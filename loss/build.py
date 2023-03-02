# encoding: utf-8

import torch.nn.functional as F
from .token_loss import TokenLoss
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy


def build_criterion(config, stage: int = 0):
    if stage == 1:
        loss = TokenLoss(config.TRAIN.SIM_GWEIGHTS)
    else:
        loss = CrossEntropyLabelSmooth()

    return loss
