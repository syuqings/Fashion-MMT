import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import framework.configbase
from framework.ops import l2norm
from modules.transformer_encoder import Encoder, VISEncoder, CrossEncoder
from modules.common import gelu


class TransformerConfig(framework.configbase.ModuleConfig):
  def __init__(self):
    super(TransformerConfig, self).__init__()
    self.vocab = 0
    self.attr_num = 2010
    self.img_max = 10
    self.src_max = 36
    self.tgt_max = 72
    self.d_model = 512
    self.n_layers = 3
    self.vis_layers = 1
    self.txt_layers = 1
    self.heads = 8
    self.dropout = 0.1
    self.encoder_sharing = False
    self.decoding = 'greedy'

    
class Transformer(nn.Module):
  def __init__(self, config):
    super(Transformer, self).__init__()
    self.config = config
    self.vis_encoder = VISEncoder(self.config.d_model, self.config.vis_layers, self.config.heads, self.config.dropout)
    self.src_encoder = Encoder(self.config.vocab, self.config.d_model, self.config.txt_layers, self.config.heads, self.config.dropout)
    self.src_encoder.pe.mode = self.vis_encoder.pe.mode

    if self.config.encoder_sharing:
      self.trg_encoder = self.src_encoder
    else:
      self.trg_encoder = Encoder(self.config.vocab, self.config.d_model, self.config.txt_layers, self.config.heads, self.config.dropout)
      self.trg_encoder.pe.mode = self.src_encoder.pe.mode
    
    self.cross_encoder = CrossEncoder(self.config.d_model, self.config.n_layers, self.config.heads, self.config.dropout)
    
    # output layers
    self.logit = nn.Linear(self.config.d_model, self.config.vocab)
    self.logit.weight = self.src_encoder.embed.embed.weight
    self.cls = nn.Linear(self.config.d_model, 2)
    self.attr_cls = nn.Linear(self.config.d_model, self.config.attr_num)
    
    self.dropout = nn.Dropout(self.config.dropout)
    self.init_weights()

  def init_weights(self,):
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)

  def forward(self, src, trg, img, src_mask, trg_mask, img_mask, task='mmt'):
    s_outputs = self.src_encoder(src, src_mask, mode=0)
    t_outputs = self.trg_encoder(trg, trg_mask, mode=1)
    i_outputs = self.vis_encoder(img, img_mask, mode=2)
    input = torch.cat([i_outputs, s_outputs, t_outputs], dim=1)

    if trg_mask is not None and trg_mask.size(1) != 1:
      firmask = torch.cat([img_mask, src_mask, trg_mask[:,0].unsqueeze(1)], dim=-1)
      firmask = firmask.repeat(1, img.size(1)+src.size(1), 1)
      img_mask = img_mask.repeat(1, trg.size(1), 1)
      src_mask = src_mask.repeat(1, trg.size(1), 1)
      secmask = torch.cat([img_mask, src_mask, trg_mask], dim=-1)
      mask = torch.cat([firmask, secmask], dim=1)
    else:
      mask = torch.cat([img_mask, src_mask, trg_mask], dim=-1)

    e_outputs = self.cross_encoder(input, mask)
    if task == 'itm':
      output = self.cls(gelu(e_outputs[:,-1]))
    elif task == 'attp':
      output = self.attr_cls(gelu(e_outputs[:,-1]))
    else:
      output = self.logit(e_outputs)
    return output

  def sample(self, src, img, src_mask, img_mask, decoding='greedy'):
    init_tok, mask_tok = 2, 4
    bs = src.size(0)
    i_outputs = self.vis_encoder(img, img_mask, mode=2)
    s_outputs = self.src_encoder(src, src_mask, mode=0)
    init_word = torch.ones(bs, 1).fill_(init_tok).long().cuda()
    trg_mask = self.nopeak_mask(1).repeat(bs, 1, 1)
    t_outputs = self.trg_encoder(init_word, trg_mask, mode=1)
    input = torch.cat([i_outputs, s_outputs, t_outputs], dim=1)
    mask = torch.cat([img_mask, src_mask, trg_mask], dim=-1)
    e_outputs = self.cross_encoder(input, mask, step=1)

    mask_word = torch.ones(bs, 1).fill_(mask_tok).long().cuda()
    outputs = torch.cat([init_word, mask_word], dim=1)
    for i in range(2, self.config.tgt_max):
      trg_mask = self.nopeak_mask(i).repeat(bs, 1, 1)
      t_outputs = self.trg_encoder(outputs, trg_mask, mode=1)
      out = self.logit(self.cross_encoder(t_outputs, trg_mask, step=i))
      logprobs = F.log_softmax(out[:,-1], dim=-1)
      if decoding == 'greedy':
        _, next_word = torch.max(logprobs, dim=1)
        next_word = next_word.unsqueeze(-1)
      else:
        probs = torch.exp(logprobs.data).cpu()
        next_word = torch.multinomial(probs, 1).cuda()
      outputs[:,-1] = next_word[:,0]
      outputs = torch.cat([outputs, mask_word], dim=1)
    return outputs

  def nopeak_mask(self, size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0).cuda()
    return np_mask

  def init_vars(self, src, img, src_mask, img_mask, beam_size):
    init_tok, mask_tok = 2, 4
    bs = src.size(0)
    i_outputs = self.vis_encoder(img, img_mask, mode=2)
    s_outputs = self.src_encoder(src, src_mask, mode=0)
    outputs = torch.LongTensor([[init_tok]] * bs).cuda()
    trg_mask = self.nopeak_mask(1).repeat(bs, 1, 1)
    t_outputs = self.trg_encoder(outputs, trg_mask, mode=1)
    input = torch.cat([i_outputs, s_outputs, t_outputs], dim=1)
    mask = torch.cat([img_mask, src_mask, trg_mask], dim=-1)
    e_outputs = self.cross_encoder(input, mask, step=1)

    mask_word = torch.ones(bs, 1).fill_(mask_tok).long().cuda()
    outputs = torch.cat([outputs, mask_word], dim=1)
    trg_mask = self.nopeak_mask(2).repeat(bs, 1, 1)
    t_outputs = self.trg_encoder(outputs, trg_mask, mode=1)
    out = self.logit(self.cross_encoder(t_outputs, trg_mask, step=2))
    out = F.softmax(out, dim=-1)
    probs, ix = out[:, -1].data.topk(beam_size)
    log_scores = torch.log(probs)
    outputs = torch.zeros(bs, beam_size, self.config.tgt_max).long().cuda()
    outputs[:, :, 0] = init_tok
    outputs[:, :, 1] = ix
    return outputs, log_scores

  def k_best_outputs(self, outputs, out, log_scores, i, k):
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1).cuda() + log_scores.transpose(0,1)
    k_probs, k_ix = log_probs.view(-1).topk(k)
    row = k_ix // k
    col = k_ix % k
    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]
    log_scores = k_probs
    return outputs, log_scores

  def beam_search(self, src, img, src_mask, img_mask, beam_size=5):
    outputs, log_scores = self.init_vars(src, img, src_mask, img_mask, beam_size)
    src_mask = src_mask.unsqueeze(1).expand(src_mask.size(0), beam_size, src_mask.size(-2), src_mask.size(-1))
    src_mask = src_mask.contiguous().view(-1, src_mask.size(-2), src_mask.size(-1))
    
    eos_tok, mask_tok = 3, 4
    bs = src.size(0)
    final = torch.zeros(bs, self.config.tgt_max).long().cuda()
    mask_word = torch.ones(1, 1).fill_(mask_tok).long().cuda()
    for i in range(2, self.config.tgt_max):
      tmp = outputs.view(-1, outputs.size(-1))[:, :i]
      tmp = torch.cat([tmp, mask_word.repeat(tmp.size(0), 1)], dim=1)
      trg_mask = self.nopeak_mask(i+1).repeat(tmp.size(0), 1, 1)
      t_outputs = self.trg_encoder(tmp, trg_mask, mode=1)
      out = self.logit(self.cross_encoder(t_outputs, trg_mask, step=i+1))
      out = F.softmax(out, dim=-1)
      out = out.view(bs, beam_size, -1, out.size(-1))
      for b in range(bs):
        outputs[b], log_scores[b] = self.k_best_outputs(outputs[b], out[b], log_scores[b].unsqueeze(0), i, beam_size) 
        ones = (outputs[b]==eos_tok).nonzero() # Occurrences of end symbols for all input sentences.
        sentence_lengths = torch.zeros(len(outputs[b]), dtype=torch.long).cuda()
        for vec in ones:
          if sentence_lengths[vec[0]]==0: # First end symbol has not been found yet
            sentence_lengths[vec[0]] = vec[1] # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])
        if num_finished_sentences == beam_size:
          alpha = 0.7
          div = 1/(sentence_lengths.type_as(log_scores[b])**alpha)
          _, ind = torch.max(log_scores[b] * div, 0)
          if final[b].sum() == 0:
            final[b] = outputs[b][ind]
    for b in range(bs):
      if final[b].sum() == 0:
        final[b] = outputs[b][0]
    return final

