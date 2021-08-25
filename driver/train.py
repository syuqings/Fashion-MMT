from __future__ import print_function
from __future__ import division

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.insert(0, os.path.abspath('..'))
import argparse
import json
import time
import pdb
import random
import numpy as np

import torch
import torch.utils.data as data

import framework.run_utils
import framework.logbase
import models.model
from models.model import DECODER
import readers.data as dataset

def set_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('model_cfg_file')
  parser.add_argument('path_cfg_file')
  parser.add_argument('--is_train', default=False, action='store_true')
  parser.add_argument('--resume_file', default=None)
  parser.add_argument('--eval_set', default='val')
  opts = parser.parse_args()

  set_seeds(12345)

  path_cfg = framework.run_utils.gen_common_pathcfg(
    opts.path_cfg_file, is_train=opts.is_train)
  if path_cfg.log_file is not None:
    _logger = framework.logbase.set_logger(path_cfg.log_file, 'trn_%f'%time.time())
  else:
    _logger = None
 
  model_cfg = models.model.TransModelConfig()
  model_cfg.load(opts.model_cfg_file)
  _model = models.model.TransModel(model_cfg, _logger=_logger)

  if opts.is_train:
    model_cfg.save(os.path.join(path_cfg.log_dir, 'model.cfg'))
    path_cfg.save(os.path.join(path_cfg.log_dir, 'path.cfg'))
    json.dump(vars(opts), open(os.path.join(path_cfg.log_dir, 'opts.cfg'), 'w'), indent=2)

    trn_reader, val_reader = {}, {}
    for task in path_cfg.pretrain_task:
      trn_data = dataset.MMTDataset(path_cfg, 'trn', model_cfg.subcfgs[DECODER].img_max, model_cfg.subcfgs[DECODER].src_max, model_cfg.subcfgs[DECODER].tgt_max, task=task, _logger=_logger)
      sampler = dataset.TokenBucketSampler(trn_data.lens, bucket_size=8192, batch_size=model_cfg.trn_batch_size, size_multiple=8)
      r = 1
      if task == 'xmlm':
        r = 9
      elif task == 'itm':
        r = 2
      trn_reader[task] = (data.DataLoader(trn_data, batch_sampler=sampler, num_workers=4), r)
    meta_loader = dataset.MetaLoader(trn_reader)

    for task in path_cfg.eval_task:
      val_data = dataset.MMTDataset(path_cfg, 'val', model_cfg.subcfgs[DECODER].img_max, model_cfg.subcfgs[DECODER].src_max, model_cfg.subcfgs[DECODER].tgt_max, task=task, _logger=_logger)
      val_reader[task] = data.DataLoader(val_data, batch_size=model_cfg.tst_batch_size, shuffle=False, num_workers=4)

    _model.train(meta_loader, val_reader, path_cfg.model_dir, path_cfg.log_dir, resume_file=opts.resume_file)

  else:
    tst_reader = {}
    for task in path_cfg.eval_task:
      tst_data = dataset.MMTDataset(path_cfg, opts.eval_set, model_cfg.subcfgs[DECODER].img_max, model_cfg.subcfgs[DECODER].src_max, model_cfg.subcfgs[DECODER].tgt_max, task=task, _logger=_logger)
      tst_reader[task] = data.DataLoader(tst_data, batch_size=model_cfg.tst_batch_size, shuffle=False, num_workers=4)

    model_str_scores = []
    is_first_eval = True
    if opts.resume_file is None:
      model_files = framework.run_utils.find_best_val_models(path_cfg.log_dir, path_cfg.model_dir)
    else:
      model_files = {'predefined': opts.resume_file}

    for measure_name, model_file in model_files.items():
      set_pred_dir = os.path.join(path_cfg.pred_dir, opts.eval_set)
      if not os.path.exists(set_pred_dir):
        os.makedirs(set_pred_dir)
      tst_pred_file = os.path.join(set_pred_dir, 
        os.path.splitext(os.path.basename(model_file))[0]+'.json')

      scores = _model.test(tst_reader, tst_pred_file, tst_model_file=model_file)
      if is_first_eval:
        score_names = scores.keys()
        model_str_scores.append(','.join(score_names))
        is_first_eval = False
        print(model_str_scores[-1])
      str_scores = [measure_name, os.path.basename(model_file)]
      for score_name in score_names:
        str_scores.append('%.4f'%(scores[score_name]))
      str_scores = ','.join(str_scores)
      print(str_scores)
      model_str_scores.append(str_scores)

    score_log_file = os.path.join(path_cfg.pred_dir, opts.eval_set, 'scores.csv')
    with open(score_log_file, 'w') as f:
      for str_scores in model_str_scores:
        print(str_scores, file=f)


if __name__ == '__main__':
  main()
