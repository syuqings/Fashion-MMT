import torch
import torch.nn as nn
from modules.common import *


class EncoderLayer(nn.Module):
  def __init__(self, d_model, heads, dropout=0.1):
    super().__init__()
    self.norm_1 = Norm(d_model)
    self.norm_2 = Norm(d_model)
    self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
    self.ff = FeedForward(d_model, dropout=dropout)
    self.dropout_1 = nn.Dropout(dropout)
    self.dropout_2 = nn.Dropout(dropout)

  def forward(self, x, mask, layer_cache=None):
    x2 = self.norm_1(x)
    x = x + self.dropout_1(self.attn(x2,x2,x2,mask, layer_cache=layer_cache))
    x2 = self.norm_2(x)
    x = x + self.dropout_2(self.ff(x2))
    return x


class Encoder(nn.Module):
  def __init__(self, vocab_size, d_model, N, heads, dropout):
    super().__init__()
    self.N = N
    self.embed = Embedder(vocab_size, d_model)
    self.pe = PositionalEncoder(d_model, dropout=dropout)
    self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
    self.norm = Norm(d_model)

  def forward(self, src, mask, mode):
    x = self.embed(src)
    if isinstance(mode, int):
      mode = torch.LongTensor([mode]).cuda()
    x = self.pe(x, mode=mode)
    for i in range(self.N):
      x = self.layers[i](x, mask)
    return self.norm(x)


class VISEncoder(nn.Module):
  def __init__(self, d_model, N, heads, dropout):
    super().__init__()
    self.N = N
    self.embed = nn.Linear(2048, d_model)
    self.pe = PositionalEncoder(d_model, dropout=dropout)
    self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
    self.norm = Norm(d_model)

  def forward(self, img, mask, mode):
    x = self.embed(img)
    x = self.pe(x, mode=mode)
    for i in range(self.N):
      x = self.layers[i](x, mask)
    return self.norm(x)


class CrossEncoder(nn.Module):
  def __init__(self, d_model, N, heads, dropout):
    super().__init__()
    self.N = N
    self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
    self.norm = Norm(d_model)
    self.cache = None

  def _init_cache(self):
    self.cache = {}
    for i in range(self.N):
      self.cache['layer_%d'%i] = {
        'self_keys': None,
        'self_values': None,
        'self_masks': None,
      }

  def forward(self, x, mask, step=None):
    if step == 1:
      self._init_cache()
    for i in range(self.N):
      layer_cache = self.cache['layer_%d'%i] if step is not None else None
      x = self.layers[i](x, mask, layer_cache=layer_cache)
    return self.norm(x)

