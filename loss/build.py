# encoding: utf-8

import torch.nn.functional as F
from .token_loss import TokenLoss, ContrastiveLoss
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy


def build_criterion(config, stage: int = 0, tokenflow=True):
    if stage == 1:
        if tokenflow:
            loss = TokenLoss(config.TRAIN.SIM_GWEIGHTS, config.TRAIN.T_GLOBAL, config.TRAIN.T_PART)
        else:
            loss = ContrastiveLoss(config.TRAIN.SIM_GWEIGHTS, config.TRAIN.T_GLOBAL, config.TRAIN.T_PART)
    else:
        loss = LabelSmoothingCrossEntropy(config.TRAIN.LABEL_SMOOTHING)

    return loss
