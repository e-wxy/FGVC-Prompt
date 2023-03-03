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

### Run locally

```
torchrun --nproc_per_node=2 train.py -n "test1" -c configs/cub.yml MODEL.PRETRAIN_FILE 'ViT-B-16.pt' MODEL.PRETRAIN_PATH './pretrained'
```

### Run on virtaicloud
```
torchrun --nproc_per_node=2 $GEMINI_RUN/Prompt/train.py \
-n "test1" -i "First Try"   \
-c $GEMINI_RUN/Prompt/configs/cub.yml   \
OUTPUT_DIR $GEMINI_DATA_OUT DATA.DATASET.ROOT_DIR $GEMINI_DATA_IN1  \
MODEL.PRETRAIN_PATH $GEMINI_PRETRAIN MODEL.PRETRAIN_FILE 'ViT-B-16.pt'
```
Dev
```
torchrun --nproc_per_node=2 $GEMINI_RUN/Prompt/train.py \
-n "test1_2" -i "Check stage 1"   \
-c $GEMINI_RUN/Prompt/configs/cub.yml   \
OUTPUT_DIR $GEMINI_DATA_OUT DATA.DATASET.ROOT_DIR $GEMINI_DATA_IN1  \
MODEL.PRETRAIN_PATH $GEMINI_PRETRAIN \
TRAIN.STAGE1.MAX_EPOCHS 5 TRAIN.STAGE2.MAX_EPOCHS 100
```

### Fine-Tune
```
torchrun --nproc_per_node=2 $GEMINI_RUN/Prompt/fine_tune.py \
-n "test2" -i "Tuning stage 2"   \
-c $GEMINI_RUN/Prompt/configs/cub.yml   \
OUTPUT_DIR $GEMINI_DATA_OUT DATA.DATASET.ROOT_DIR $GEMINI_DATA_IN1  \
MODEL.PRETRAIN_PATH $GEMINI_PRETRAIN
```
Dev
```
torchrun --nproc_per_node=2 $GEMINI_RUN/Prompt/fine_tune.py \
-n "test3" -i "Tuning stage 2"   \
-c $GEMINI_RUN/Prompt/configs/cub.yml   \
OUTPUT_DIR $GEMINI_DATA_OUT DATA.DATASET.ROOT_DIR $GEMINI_DATA_IN1  \
MODEL.PRETRAIN_PATH $GEMINI_DATA_OUT
```

## To Tune

### Hyper-Params for Prompting

1. Dropout rate in text description: `DATA.DATASET.DROP_RATE`
2. Temperature in TokenFlow: `MODEL.LAMB`

### Classifier



## Acknowledgement

Codebase from [CLIP](https://github.com/openai/CLIP), [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
