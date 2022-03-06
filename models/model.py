from __future__ import print_function
from __future__ import division

import numpy as np
import json
import pdb
from tqdm import tqdm
import time
import io , sys
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import framework.configbase
import framework.modelbase
import modules.transformer
import utils.evaluation
import utils.criterion

DECODER = 'transformer'

class TransModelConfig(framework.configbase.ModelConfig):
  def __init__(self):
    super(TransModelConfig, self).__init__()

  def load(self, cfg_file):
    with open(cfg_file) as f:
      data = json.load(f)
    for key, value in data.items():
      if key != 'subcfgs':
        setattr(self, key, value)
    # initialize config objects
    for subname, subcfg_type in self.subcfg_types.items():
      if subname == DECODER:
        self.subcfgs[subname] = modules.transformer.__dict__[subcfg_type]()
      self.subcfgs[subname].load_from_dict(data['subcfgs'][subname])
      
class TransModel(framework.modelbase.ModelBase):
  def build_submods(self):
    submods = {}
    submods[DECODER] = modules.transformer.Transformer(self.config.subcfgs[DECODER])
    return submods

  def build_loss(self):
    xe = utils.criterion.LabelSmoothingLoss(0.1,self.config.subcfgs[DECODER].vocab,1)
    classify = nn.CrossEntropyLoss()
    multilabel = utils.criterion.MultilabelCategoricalLoss()
    return (xe, classify, multilabel)

  def forward_loss(self, batch_data, task='mmt', step=None):
    src = batch_data['src_ids'].cuda()
    trg = batch_data['trg_ids'].cuda()
    src_mask, trg_mask = self.create_masks(src, trg, task)
    img_ft = batch_data['img_ft'].cuda()
    img_len = batch_data['ft_len'].cuda()
    img_mask = self.img_mask(img_len, max_len=img_ft.size(1)).unsqueeze(1)
    outputs = self.submods[DECODER](src, trg, img_ft, src_mask, trg_mask, img_mask, task=task)

    if task == 'itm':
      loss = self.criterion[1](outputs, batch_data['align_label'].cuda())
    elif task == 'attp':
      loss = self.criterion[2](outputs, batch_data['attr_label'].float().cuda())
    else:
      outputs = nn.LogSoftmax(dim=-1)(outputs[:,img_ft.size(1):])
      output_label = batch_data['output_label'].cuda()
      ys = output_label.contiguous().view(-1)
      norm = output_label.ne(1).sum().item()
      loss = self.criterion[0](outputs.view(-1, outputs.size(-1)), ys, norm)
    return loss

  def evaluate(self, tst_reader):
    pred_sents, ref_sents = [], []
    attr_pred, attr_label = [], []
    score = {}
    n_correct, n_word = 0, 0
    for task in tst_reader:
      cur_reader = tst_reader[task]
      for batch_data in tqdm(cur_reader):
        src = batch_data['src_ids'].cuda()
        trg = batch_data['trg_ids'].cuda()
        src_mask, trg_mask = self.create_masks(src, trg, task)
        img_ft = batch_data['img_ft'].cuda()
        img_len = batch_data['ft_len'].cuda()
        img_mask = self.img_mask(img_len, max_len=img_ft.size(1)).unsqueeze(1)

        if task == 'mmt':
          if self.submods[DECODER].config.decoding == 'greedy':
            output = self.submods[DECODER].sample(src, img_ft, src_mask, img_mask)
          else:
            output = self.submods[DECODER].beam_search(src, img_ft, src_mask, img_mask)
          translations = cur_reader.dataset.int2sent(output.detach())
          ref_sents.extend(batch_data['ref_sents'])
          pred_sents.extend(translations)
        elif task == 'itm':
          target = batch_data['align_label'].cuda()
          output = self.submods[DECODER](src, trg, img_ft, src_mask, trg_mask, img_mask, task=task)
          pred = output.max(1, keepdim=True)[1]
          n_correct += float(pred.eq(target.view_as(pred)).cpu().float().sum())
          n_word += output.size(0)
        elif task == 'attp':
          output = self.submods[DECODER](src, trg, img_ft, src_mask, trg_mask, img_mask, task=task)
          attr_pred.extend(output.detach().cpu().numpy())
          attr_label.extend(batch_data['attr_label'].detach().numpy())
        else:
          output_label = batch_data['output_label'].cuda()
          output = self.submods[DECODER](src, trg, img_ft, src_mask, trg_mask, img_mask, task=task)[:,img_ft.size(1):]
          output = output[output_label != 1]
          output_label = output_label[output_label != 1]
          n_correct += (output.max(dim=-1)[1] == output_label).sum().item()
          n_word += output_label.numel()

      if task == 'mmt':
        score.update(utils.evaluation.compute(pred_sents, ref_sents))
      elif task == 'attp':
        r_1, r_5, p_1, p_5 = utils.evaluation.compute_multilabel(np.array(attr_pred), np.array(attr_label))
        score.update({task+'_r@1':r_1, task+'_r@5':r_5, task+'_p@1':p_1, task+'_p@5':p_5})
      else:
        score.update({task+'_avg_acc':n_correct/n_word})
    return score, pred_sents

  def validate(self, val_reader):
    self.eval_start()
    metrics, _ = self.evaluate(val_reader)
    return metrics

  def test(self, tst_reader, tst_pred_file, tst_model_file=None):
    if tst_model_file is not None:
      self.load_checkpoint(tst_model_file)
    self.eval_start()
    metrics, pred_data = self.evaluate(tst_reader)
    with open(tst_pred_file, 'w') as f:
      json.dump(pred_data, f)
    return metrics

  def nopeak_mask(self, size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0).cuda()
    return np_mask

  def create_masks(self, src, trg=None, task='mmt'):
    src_mask = (src != 1).unsqueeze(-2)   # 1 is src_pad, trg_pad
    if trg is not None:
      trg_mask = (trg != 1).unsqueeze(-2)
      if task == 'mmt':
        size = trg.size(1) # get seq_len for matrix
        np_mask = self.nopeak_mask(size)
        trg_mask = trg_mask & np_mask
    else:
      trg_mask = None
    return src_mask, trg_mask

  def img_mask(self, lengths, max_len=None):
    ''' Creates a boolean mask from sequence lengths.
        lengths: LongTensor, (batch, )
    '''
    batch_size = lengths.size(0)
    max_len = max_len or lengths.max()
    return ~(torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .ge(lengths.unsqueeze(1)))