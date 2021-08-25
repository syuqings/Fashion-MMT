import torch
import torch.nn as nn
import torch.nn.functional as F

import framework.configbase
import framework.ops
import pdb
import numpy as np


class LabelSmoothingLoss(nn.Module):
  """
  With label smoothing,
  KL-divergence between q_{smoothed ground truth prob.}(w)
  and p_{prob. computed by model}(w) is minimized.
  """
  def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
    assert 0.0 < label_smoothing <= 1.0
    self.padding_idx = ignore_index
    super(LabelSmoothingLoss, self).__init__()
    smoothing_value = label_smoothing / (tgt_vocab_size - 2)
    one_hot = torch.full((tgt_vocab_size,), smoothing_value).cuda()
    one_hot[self.padding_idx] = 0
    self.register_buffer('one_hot', one_hot.unsqueeze(0))
    self.confidence = 1.0 - label_smoothing

  def forward(self, output, target, norm):
    """
    output (FloatTensor): batch_size x n_classes
    target (LongTensor): batch_size
    """
    model_prob = self.one_hot.repeat(target.size(0), 1)
    model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
    model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)
    loss = F.kl_div(output, model_prob, reduction='sum')
    return loss.div(float(norm))
    

class MultilabelCategoricalLoss(nn.Module):
  def __init__(self):
    super(MultilabelCategoricalLoss,self).__init__()

  def forward(self, y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

