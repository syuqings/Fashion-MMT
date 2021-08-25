import numpy as np
import torch
from cap_eval.bleu.bleu import Bleu
from cap_eval.meteor.meteor import Meteor
from cap_eval.cider.cider import Cider

bleu_scorer = Bleu(4)
cider_scorer = Cider()
meteor_scorer = Meteor()

def bleu_eval(refs, cands):
  print ("calculating bleu_4 score...")
  bleu, _ = bleu_scorer.compute_score(refs, cands)
  return bleu

def cider_eval(refs, cands):
  print ("calculating cider score...")
  cider, _ = cider_scorer.compute_score(refs, cands)
  return cider

def meteor_eval(refs, cands):
  print ("calculating meteor score...")
  meteor, _ = meteor_scorer.compute_score(refs, cands)
  return meteor

def compute(preds, refs):
  refcaps = {}
  candcaps = {}
  for i in range(len(preds)):
    candcaps[str(i)] = [preds[i]]
    refcaps[str(i)] = [refs[i]]

  bleu = bleu_eval(refcaps, candcaps)
  cider = cider_eval(refcaps, candcaps)
  meteor = meteor_eval(refcaps, candcaps)
  
  scores = {'cider': cider,
            'meteor': meteor,
            'bleu_4': bleu[3],
            'bleu_3': bleu[2],
            'bleu_2': bleu[1],
            'bleu_1':bleu[0]
           }
  return scores

def compute_multilabel(attr_prob, attr_vector):
  sorted_idxs = np.argsort(-attr_prob)[:,:5]
  count_r, correct_1, correct_5 = 0, 0, 0

  for i in range(len(sorted_idxs)):
    attr_label = np.where(attr_vector[i]==1)[0]
    count_r += len(attr_label)
    for j in range(len(attr_label)):
      if attr_label[j] == sorted_idxs[i][0]:
        correct_1 += 1
      if attr_label[j] in sorted_idxs[i]:
        correct_5 += 1

  r_1 = correct_1 / count_r
  r_5 = correct_5 / count_r
  p_1 = correct_1 / len(attr_prob)
  p_5 = correct_5 / (len(attr_prob)*5)
  return r_1, r_5, p_1, p_5

