from __future__ import print_function
from __future__ import division

import os
import json
import numpy as np
import random
import pdb
import math

from cytoolz import partition_all
import torch.utils.data
from torch.utils.data import Sampler

UNK, PAD, BOS, EOS, MASK = 0, 1, 2, 3, 4


class MMTDataset(torch.utils.data.Dataset):
  def __init__(self, config, split, img_max=10, src_max=36, tgt_max=72, task='mmt', _logger=None):
    super(MMTDataset, self).__init__()

    if _logger is None:
      self.print_fn = print
    else:
      self.print_fn = _logger.info

    self.names = np.load(config.name_file[split])
    self.anno = json.load(open(config.anno_file))
    self.src = open(config.src_txt[split], 'r', encoding='utf-8').readlines()
    self.trg = open(config.tgt_txt[split], 'r', encoding='utf-8').readlines()
    self.num_text = len(self.src)
    self.print_fn('text size %d' % self.num_text)

    self.lens = []
    if task == 'xmlm':
      for i in range(len(self.trg)):
        self.lens.append((len(self.trg[i].strip().split())+len(self.src[i].strip().split())+2)/2)
    elif task == 'mmt':
      for i in range(len(self.trg)):
        self.lens.append(len(self.trg[i].strip().split())+2)
    elif task in ['attp', 'itm']:
      for i in range(len(self.src)):
        self.lens.append(len(self.src[i].strip().split())+2)
    
    self.sim_img = json.load(open(config.sim_img[split]))
    self.stoi = json.load(open(config.word2int_file))
    self.itos = json.load(open(config.int2word_file))
    self.atoi = json.load(open(config.attr2int_file))
    self.ft_root = config.ft_root
    self.img_max = img_max
    self.src_max = src_max
    self.tgt_max = tgt_max
    self.is_train = True if split == 'trn' else False
    self.task = task

  def mask_and_pad_sent(self, x, id=None, lang='src'):
    max_len = self.src_max if lang == 'src' else self.tgt_max

    # masking input sequence
    if self.task == 'xmlm' or (self.task == 'mmt' and lang == 'trg'):  # cross-lingual masking or adapt to MMT
      x, output_label = self.mask_sent(x[:max_len-1])
    elif self.task == 'attp':
      x, output_label = self.get_attr(x[:max_len-1], id)
    else:
      output_label = [PAD] * (max_len-1)

    # padding input sequence
    prob = random.random()
    if self.task == 'mmt' and lang == 'trg' and prob < 0.12:
      padded = [BOS] + x[:max_len-1] + [MASK] + [PAD] * max(0, max_len - len(x) - 2)
      output_label = [PAD] + output_label + [EOS] + [PAD] * max(0, max_len - len(x) - 2)
    elif self.task == 'attp':
      padded = [BOS] + x[:max_len-1] + [EOS] + [PAD] * max(0, max_len - len(x) - 2)
    else:
      padded = [BOS] + x[:max_len-1] + [EOS] + [PAD] * max(0, max_len - len(x) - 2)
      output_label = [PAD] + output_label + [PAD] + [PAD] * max(0, max_len - len(x) - 2)

    # truncate with the max length
    length = min(len(x)+2, max_len)
    padded = padded[:max_len]
    if self.task != 'attp':
      output_label = output_label[:max_len]
    return np.array(padded), np.array(output_label), length

  def random_mask(self, x, i, prob):
    # 80% randomly change token to mask token
    if prob < 0.8:
      x[i] = MASK
    # 10% randomly change token to random token
    elif prob < 0.9:
      x[i] = random.choice(list(range(len(self.stoi))))
    # -> rest 10% randomly keep current token
    return x

  def mask_sent(self, x):
    output_label = []
    for i, token in enumerate(x):
      prob = random.random()
      # mask normal token with 15% probability
      if prob < 0.15:
        prob /= 0.15
        x = self.random_mask(x, i, prob)
        output_label.append(token)
      else:
        # no masking token (will be ignored by loss function later)
        output_label.append(PAD)
    return x, output_label

  def get_attr(self, x, id):
    attrs = []
    output_label = [0.] * len(self.atoi)
    for attr in self.anno[id]['attr']:
      try:
        output_label[self.atoi[attr]] = 1.
        prob = random.random()
        if self.stoi[attr] in x:
          x = self.random_mask(x, x.index(self.stoi[attr]), prob)
        elif self.stoi[attr+'s'] in x:
          x = self.random_mask(x, x.index(self.stoi[attr+'s']), prob)
        elif self.stoi[attr+'es'] in x:
          x = self.random_mask(x, x.index(self.stoi[attr+'es']), prob)
      except:
        pass
    return x, output_label

  def sent2int(self, str_sent):
    int_sent = [self.stoi.get(w, UNK) for w in str_sent.split()]
    return int_sent

  def int2sent(self, batch):
    with torch.cuda.device_of(batch):
      batch = batch.tolist()
    batch = [[self.itos.get(str(ind), '<unk>') for ind in ex] for ex in batch] # denumericalize
    
    def trim(s, t):
      sentence = []
      for w in s:
        if w == t:
          break
        sentence.append(w)
      return sentence
    batch = [trim(ex, '<eos>') for ex in batch] # trim past frst eos

    def filter_special(tok):
      return tok not in ('<sos>', '<pad>', '<mask>')
    batch = [" ".join(filter(filter_special, ex)).replace("@@ ", "") for ex in batch]
    return batch

  def __len__(self):
    return self.num_text

  def __getitem__(self, idx):
    outs = {}
    name = self.names[idx]
    img_ft = np.zeros(shape=[self.img_max, 2048], dtype=np.float32)
    for i, img in enumerate(self.anno[name]['images']):
      if i >= self.img_max:
        break
      img_ft[i] = np.load(os.path.join(self.ft_root, img+".npy"))[0]
    ft_len = min(self.img_max, len(self.anno[name]['images']))

    if self.task in ['xmlm', 'mmt']:
      src_id, src_label, src_len = self.mask_and_pad_sent(self.sent2int(self.src[idx].strip()), id=name, lang='src')
      trg_id, trg_label, trg_len = self.mask_and_pad_sent(self.sent2int(self.trg[idx].strip()), id=name, lang='trg')
    elif self.task == 'attp':
      src_id, src_label, src_len = self.mask_and_pad_sent(self.sent2int(self.src[idx].strip()), id=name, lang='src')
      trg_id = np.array([BOS])
      trg_len = 1
    elif self.task == 'itm':
      rep_prob = random.random()
      if rep_prob < 0.5:
        old_idx = idx
        idx = random.choice(self.sim_img[self.anno[name]['category']])
        if old_idx == idx:
          align_label = 1
        else:
          align_label = 0
      else:
        align_label = 1
      src_id, src_label, src_len = self.mask_and_pad_sent(self.sent2int(self.src[idx].strip()), id=name, lang='src')
      trg_id = np.array([BOS])
      trg_len = 1

    outs['ft_len'] = ft_len
    outs['img_ft'] = img_ft
    outs['src_ids'] = src_id
    outs['src_lens'] = src_len
    outs['trg_ids'] = trg_id
    outs['trg_lens'] = trg_len
    outs['ref_sents'] = self.trg[idx].strip()
    if self.task == 'itm':
      outs['align_label'] = align_label
    elif self.task == 'attp':
      outs['attr_label'] = src_label
    else:
      outs['output_label'] = np.concatenate([src_label, trg_label], axis=0)
    return outs


class TokenBucketSampler(Sampler):
  def __init__(self, lens, bucket_size, batch_size, droplast=False, size_multiple=8):
    self._lens = lens
    self._max_tok = batch_size
    self._bucket_size = bucket_size
    self._droplast = droplast
    self._size_mul = size_multiple

  def _create_ids(self):
    return list(range(len(self._lens)))

  def _sort_fn(self, i):
    return self._lens[i]

  def __iter__(self):
    ids = self._create_ids()
    random.shuffle(ids)
    buckets = [sorted(ids[i:i+self._bucket_size], key=self._sort_fn, reverse=True)
                for i in range(0, len(ids), self._bucket_size)]
    # fill batches until max_token (include padding)
    batches = []
    for bucket in buckets:
      max_len = 0
      batch_indices = []
      for indices in partition_all(self._size_mul, bucket):
        max_len = max(max_len, max(self._lens[i] for i in indices))
        if (max_len * (len(batch_indices) + self._size_mul)
          > self._max_tok):
          if not batch_indices:
            raise ValueError("max_tokens too small / max_seq_len too long")
          assert len(batch_indices) % self._size_mul == 0
          batches.append(batch_indices)
          batch_indices = list(indices)
          max_len = max(self._lens[i] for i in indices)
        else:
          batch_indices.extend(indices)
      if not self._droplast and batch_indices:
        batches.append(batch_indices)
    random.shuffle(batches)
    return iter(batches)

  def __len__(self):
    raise ValueError("NOT supported. ")


class MetaLoader(object):
  """ wraps multiple data loaders """
  def __init__(self, loaders, accum_steps=1):
    assert isinstance(loaders, dict)
    self.name2loader = {}
    self.name2iter = {}
    self.sampling_pools = []
    for n, l in loaders.items():
      if isinstance(l, tuple):
        l, r = l
      elif isinstance(l, torch.utils.data.DataLoader):
        r = 1
      else:
        raise ValueError()
      self.name2loader[n] = l
      self.name2iter[n] = iter(l)
      self.sampling_pools.extend([n]*r)
    self.accum_steps = accum_steps
    self.step = 0

  def __iter__(self):
    """ this iterator will run indefinitely """
    task = self.sampling_pools[0]
    while True:
      if self.step % self.accum_steps == 0:
        task = random.choice(self.sampling_pools)
        self.step += 1
        iter_ = self.name2iter[task]
        try:
          batch = next(iter_)
        except StopIteration:
          iter_ = iter(self.name2loader[task])
          batch = next(iter_)
          self.name2iter[task] = iter_

      yield task, batch
     


