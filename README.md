# Prompting Attributes for FGVC

## Prerequisites
```
pytorch
torchvision
timm
yacs
regex
ftfy
tqdm
```

## Dataset

[Caltech-UCSD Birds-200-2011 (CUB-200-2011)](https://www.kaggle.com/datasets/wenewone/cub2002011)

## Train

Run locally

```
torchrun --nproc_per_node=2 train.py -n "test1" -c configs/cub.yml MODEL.PRETRAIN_FILE 'ViT-B-16.pt' MODEL.PRETRAIN_PATH './pretrained'
```

Run on virtaicloud
```
torchrun --nproc_per_node=2 $GEMINI_RUN/Prompt/train.py \
-n "test1" -i "First Try"   \
-c $GEMINI_RUN/Prompt/configs/cub.yml   \
OUTPUT_DIR $GEMINI_DATA_OUT DATA.DATASET.ROOT_DIR $GEMINI_DATA_IN1  \
MODEL.PRETRAIN_PATH $GEMINI_PRETRAIN MODEL.PRETRAIN_FILE 'ViT-B-16.pt'
```

Fine-Tune
```
torchrun --nproc_per_node=2 $GEMINI_RUN/Prompt/train.py \
-n "test2" -i "Tuning stage 2"   \
-c $GEMINI_RUN/Prompt/configs/cub.yml   \
OUTPUT_DIR $GEMINI_DATA_OUT DATA.DATASET.ROOT_DIR $GEMINI_DATA_IN1  \
MODEL.PRETRAIN_PATH $GEMINI_PRETRAIN MODEL.PRETRAIN_FILE 'model/pair.pt'
```


## Acknowledgement

Codebase from [CoOp](https://github.com/KaiyangZhou/CoOp)
