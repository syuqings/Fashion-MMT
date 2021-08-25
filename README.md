# Product-oriented Machine Translation with Cross-modal Cross-lingual Pre-training
This repository contains the Fashion-MMT dataset and PyTorch implementation of our paper [Product-oriented Machine Translation with Cross-modal Cross-lingual Pre-training]() (ACMMM 2021 Oral).

## Requirements
- Python 3.6
- Java 15.0.2
- PyTorch 1.1
- numpy, tqdm, h5py, scipy, six

## Fashion-MMT Dataset
### Annotations
Annotations of Fashion-MMT(C) and Fashion-MMT(L) datasets can be downloaded from [BaiduNetdisk](https://pan.baidu.com/s/1ouGADKTOyy35SBele2dWPw) (code: i55n).
```
JSON Format:
[
  {
    "id": int,
    "split": str,
    "en": str,
    "zh": str,
    "images": [str],
    "category": str,
    "attr": [str]
  }
]
```
The images can be downloaded from the url https://n.nordstrommedia.com/id/sr3/image_name with image_names.

### Features
We also provide the image features of ResNet101 pretrained on ImageNet and finetuned on Fashion-MMT at [BaiduNetdisk](https://pan.baidu.com/s/1ouGADKTOyy35SBele2dWPw) (code: i55n)（~57G）. 
Decompress and merge the downloaded features into one folder:
```bash
$ cat resnet101.finetune.tar.gz* | tar -xz
```

## Training & Inference
### Start training
1. Pre-train the model with three pre-training tasks
```bash
$ CUDA_VISIBLE_DEVICES=0 python train.py ../results/pretrain/model.json ../results/pretrain/path.json --is_train
```
2. Fine-tune the model to MMT
```bash
$ CUDA_VISIBLE_DEVICES=0 python train.py ../results/finetune/model.json ../results/finetune/path.json --is_train --resume_file ../results/pretrain/model/step.*.th
```

### Evaluation
```bash
$ CUDA_VISIBLE_DEVICES=0 python train.py ../results/finetune/model.json ../results/finetune/path.json --eval_set val --resume_file ../results/finetune/model/step.*.th
```

## Reference
If you find this repo helpful, please consider citing:
```
@inproceedings{song2021FashionMMT,
  title={Product-oriented Machine Translation with Cross-modal Cross-lingual Pre-training},
  author={Song, Yuqing and Chen, Shizhe and Jin, Qin and Luo, Wei and Xie, Jun and Huang, Fei},
  booktitle={Proceedings of the 29th {ACM} International Conference on Multimedia},
  year={2021}
}
```

