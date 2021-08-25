import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class Embedder(nn.Module):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embed = nn.Embedding(vocab_size, d_model)
  def forward(self, x):
    return self.embed(x)


class PositionalEncoder(nn.Module):
  def __init__(self, d_model, max_seq_len=200, dropout=0.1):
    super().__init__()
    self.d_model = d_model
    self.dropout = nn.Dropout(dropout)
    self.mode = nn.Embedding(3, d_model)
    # create constant 'pe' matrix with values dependant on pos and i
    pe = torch.zeros(max_seq_len, d_model)
    for pos in range(max_seq_len):
      for i in range(0, d_model, 2):
        pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
        pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)
  
  def forward(self, x, mode=None):
    # make embeddings relatively larger
    x = x * math.sqrt(self.d_model)
    #add constant to embedding
    if isinstance(mode, int):
      x = x + self.mode(torch.LongTensor([mode]).cuda()).unsqueeze(1)
    else:
      seq_len = x.size(1)
      pe = Variable(self.pe[:,:seq_len], requires_grad=False).cuda()
      if mode is not None:
        x = x + pe + self.mode(mode).unsqueeze(1)
    return self.dropout(x)


class Norm(nn.Module):
  def __init__(self, d_model, eps=1e-6):
    super().__init__()
    self.size = d_model   
    # create two learnable parameters to calibrate normalisation
    self.alpha = nn.Parameter(torch.ones(self.size))
    self.bias = nn.Parameter(torch.zeros(self.size))
    self.eps = eps
    
  def forward(self, x):
    norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
      / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
    return norm


def attention(q, k, v, d_k, mask=None, dropout=None):
  scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
  if mask is not None:
    mask = mask.unsqueeze(1)
    scores = scores.masked_fill(mask == 0, -1e9)
  scores = F.softmax(scores, dim=-1) 
  if dropout is not None:
    scores = dropout(scores)     
  output = torch.matmul(scores, v)
  return output

class MultiHeadAttention(nn.Module):
  def __init__(self, heads, d_model, dropout=0.1):
    super().__init__() 
    self.d_model = d_model
    self.d_k = d_model // heads
    self.h = heads
    self.q_linear = nn.Linear(d_model, d_model)
    self.v_linear = nn.Linear(d_model, d_model)
    self.k_linear = nn.Linear(d_model, d_model)  
    self.dropout = nn.Dropout(dropout)
    self.out = nn.Linear(d_model, d_model)

  def shape(self, x):
    bs = x.size(0)
    return x.view(bs, -1, self.h, self.d_k).transpose(1,2)
    
  def forward(self, q, k, v, mask=None, layer_cache=None):
    if layer_cache is not None:
      k = self.shape(self.k_linear(k))
      v = self.shape(self.v_linear(v))
      if layer_cache['self_keys'] is not None:
        if layer_cache['self_keys'].size(0) != k.size(0):
          beam_size = k.size(0) // layer_cache['self_keys'].size(0)
          context = layer_cache['self_keys']
          context = context.unsqueeze(1).expand(context.size(0), beam_size, context.size(-3), context.size(-2), context.size(-1))
          layer_cache['self_keys'] = context.contiguous().view(-1, context.size(-3), context.size(-2), context.size(-1))
        k = torch.cat((layer_cache['self_keys'], k), dim=2)
      else:
        layer_cache['self_keys'] = k[:,:,:-1]

      if layer_cache['self_values'] is not None:
        if layer_cache['self_values'].size(0) != v.size(0):
          beam_size = v.size(0) // layer_cache['self_values'].size(0)
          context = layer_cache['self_values']
          context = context.unsqueeze(1).expand(context.size(0), beam_size, context.size(-3), context.size(-2), context.size(-1))
          layer_cache['self_values'] = context.contiguous().view(-1, context.size(-3), context.size(-2), context.size(-1))
        v = torch.cat((layer_cache['self_values'], v), dim=2)
      else:
        layer_cache['self_values'] = v[:,:,:-1]
        layer_cache['self_masks'] = mask[:,:,:-1]
    else:
      k = self.shape(self.k_linear(k))
      v = self.shape(self.v_linear(v))

    bs = q.size(0) 
    q =  self.shape(self.q_linear(q))
    if mask.size(-1) != k.size(-2):
      if layer_cache['self_masks'].size(0) != mask.size(0):
        beam_size = mask.size(0) // layer_cache['self_masks'].size(0)
        context = layer_cache['self_masks']
        context = context.unsqueeze(1).expand(context.size(0), beam_size, context.size(-2), context.size(-1))
        layer_cache['self_masks'] = context.contiguous().view(-1, context.size(-2), context.size(-1))
      mask = torch.cat([layer_cache['self_masks'].repeat(1,mask.size(1),1), mask], dim=-1)
    # calculate attention using function we will define next
    scores = attention(q, k, v, self.d_k, mask, self.dropout)
    # concatenate heads and put through final linear layer
    concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
    output = self.out(concat)
    return output


def gelu(x):
  return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))

class FeedForward(nn.Module):
  def __init__(self, d_model, d_ff=2048, dropout=0.1):
    super().__init__() 
    # We set d_ff as a default to 2048
    self.linear_1 = nn.Linear(d_model, d_ff)
    self.dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(d_ff, d_model)
    
  def forward(self, x):
    #x = self.dropout(F.relu(self.linear_1(x), inplace=True))
    x = self.dropout(gelu(self.linear_1(x)))
    x = self.linear_2(x)
    return x


class MultiModalFusion(nn.Module):
  def __init__(self, d_model, heads, dropout=0.1):
    super().__init__()
    self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
    self.linear_q1 = nn.Linear(d_model, d_model, bias=False)
    self.linear_q2 = nn.Linear(d_model, d_model, bias=False)
    self.linear_v1 = nn.Linear(d_model, d_model, bias=True)
    self.linear_v2 = nn.Linear(d_model, d_model, bias=True)
    self.norm_1 = Norm(d_model)
    self.norm_2 = Norm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, v, vmask):
    v2 = self.norm_1(self.dropout(self.attn(x, v, v, vmask)))
    ctx = torch.tanh(self.linear_q1(x)+self.linear_v1(v2))
    gate = torch.sigmoid(self.linear_q2(x)+self.linear_v2(v2))
    f = gate * x + (1 - gate) * ctx
    return self.norm_2(f)


def get_clones(module, N):
  return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
