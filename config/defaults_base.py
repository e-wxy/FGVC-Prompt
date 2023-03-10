from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config Definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model name
_C.MODEL.NAME = 'CLIP-Attri'
# Model backbone
_C.MODEL.BACKBONE_NAME = 'ViT-B/16'
# Directory of pretrained models
_C.MODEL.PRETRAIN_PATH = './pretrained'
# Filename of pretrained model
_C.MODEL.PRETRAIN_FILE = ''
# Class number
_C.MODEL.CLASS_NUM = 300
# Hidden layer
_C.MODEL.HIDDEN_DIM = 512
# Temperature for TokenFlow
_C.MODEL.LAMB = 1.0


# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------
_C.DATA = CN()
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATA.DATASET = CN()
_C.DATA.DATASET.NAME = 'cub'
_C.DATA.DATASET.ROOT_DIR = './datasets'
_C.DATA.DATASET.DROP_RATE = 0.4

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATA.DATALOADER = CN()
_C.DATA.DATALOADER.NUM_WORKERS = 8
_C.DATA.DATALOADER.BATCH_SIZE = 16
_C.DATA.DATALOADER.TEST_BATCH_SIZE = 128
_C.DATA.DATALOADER.TEST_CROP = True
_C.DATA.DATALOADER.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.DATALOADER.INTERPOLATION = 'bicubic'
_C.DATA.DATALOADER.TRAIN_INTERPOLATION = 'bicubic'
# Sampler for data loading
_C.DATA.DATALOADER.SAMPLER = 'softmax'
_C.DATA.DATALOADER.PIN_MEMORY = True


# -----------------------------------------------------------------------------
# Augmentation
# -----------------------------------------------------------------------------
_C.DATA.AUG = CN()
# Color jitter factor
_C.DATA.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.DATA.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.DATA.AUG.REPROB = 0.25
# Random erase mode
_C.DATA.AUG.REMODE = 'pixel'
# Random erase count
_C.DATA.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.DATA.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.DATA.AUG.CUTMIX = 1.0
# # Cutmix min/max ratio, overrides alpha and enables cutmix if set
# _C.DATA.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.DATA.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.DATA.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.DATA.AUG.MIXUP_MODE = 'batch'


# -----------------------------------------------------------------------------
# TRAIN
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.SEED = 123
# Weights for similarity: 
# loss = loss_g * SIM_GWEIGHTS + (loss_v + loss_t) * (1 - SIM_GWEIGHTS) / 2
_C.TRAIN.SIM_GWEIGHTS = 0.5
_C.TRAIN.T_GLOBAL = 0.07
_C.TRAIN.T_PART = 0.07
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient Accumulation
# IMS_PER_BATCH = ACCUMULATION_STEPS * DATALOADER.BATCH_SIZE
_C.TRAIN.IMS_PER_BATCH = 16
_C.TRAIN.ACCUMULATION_STEPS = 1

# OTHER TRICKS
# Automatic Mixed Precision Training
_C.TRAIN.PREC = "amp"
# Clip Gradient Norm
_C.TRAIN.CLIP_GRAD = 1.0
_C.TRAIN.LABEL_SMOOTHING = 0.1


# -----------------------------------------------------------------------------
# STAGE 1
# -----------------------------------------------------------------------------
_C.TRAIN.STAGE1 = CN()

_C.TRAIN.STAGE1.START_EPOCH = 0
_C.TRAIN.STAGE1.MAX_EPOCHS = 200
_C.TRAIN.STAGE1.WARMUP_EPOCHS = 20
_C.TRAIN.STAGE1.WEIGHT_DECAY = 1e-4
_C.TRAIN.STAGE1.BASE_LR = 5e-4
_C.TRAIN.STAGE1.WARMUP_LR = 5e-7
_C.TRAIN.STAGE1.MIN_LR = 5e-6
# Scheduler
_C.TRAIN.STAGE1.SCHEDULER = CN()
_C.TRAIN.STAGE1.SCHEDULER.NAME = 'cosine'
# # scheduler(optimizer, **params)
# _C.TRAIN.STAGE1.SCHEDULER.PARAMS = {}
# Optimizer
_C.TRAIN.STAGE1.OPTIMIZER = CN()
_C.TRAIN.STAGE1.OPTIMIZER.NAME = 'AdamW'
# optimizer(model.parameters(), **params)
_C.TRAIN.STAGE1.OPTIMIZER.PARAMS = "lr = 5e-4; weight_decay = 1e-4; eps = 1e-8; betas = (0.9, 0.999)"
# _C.TRAIN.STAGE1.OPTIMIZER.PARAMS = "{'lr': 5e-4, 'weight_decay': 1e-4, 'momentum': 0.9, 'nesterov': True}"

# Intervals of checkpoint, log, eval
_C.TRAIN.STAGE1.LOG_PERIOD = 1
_C.TRAIN.STAGE1.EVAL_PERIOD = 5
_C.TRAIN.STAGE1.CHECKPOINT_PERIOD = 10
_C.TRAIN.STAGE1.USE_CHECKPOINT = False


# -----------------------------------------------------------------------------
# STAGE 2
# -----------------------------------------------------------------------------
_C.TRAIN.STAGE2 = CN()
_C.TRAIN.STAGE2.START_EPOCH = 0
_C.TRAIN.STAGE2.MAX_EPOCHS = 100
_C.TRAIN.STAGE2.WARMUP_EPOCHS = 10
_C.TRAIN.STAGE2.WEIGHT_DECAY = 1e-4
_C.TRAIN.STAGE2.BASE_LR = 5e-4
_C.TRAIN.STAGE2.WARMUP_LR = 5e-7
_C.TRAIN.STAGE2.MIN_LR = 5e-6
# Scheduler
_C.TRAIN.STAGE2.SCHEDULER = CN()
_C.TRAIN.STAGE2.SCHEDULER.NAME = 'cosine'
# # scheduler(optimizer, **params)
# _C.TRAIN.STAGE2.SCHEDULER.PARAMS = {}
# Optimizer
_C.TRAIN.STAGE2.OPTIMIZER = CN()
_C.TRAIN.STAGE2.OPTIMIZER.NAME = 'AdamW'
# optimizer(model.parameters(), **params)
_C.TRAIN.STAGE2.OPTIMIZER.PARAMS = "lr = 5e-4; weight_decay = 1e-4; eps = 1e-8; betas = (0.9, 0.999)"
# _C.TRAIN.STAGE2.OPTIMIZER.PARAMS = "lr = 5e-4, weight_decay = 1e-4, eps = 1e-8, betas = (0.9, 0.999)"

# Intervals of checkpoint, log, eval
_C.TRAIN.STAGE2.LOG_PERIOD = 1
_C.TRAIN.STAGE2.EVAL_PERIOD = 5
_C.TRAIN.STAGE2.CHECKPOINT_PERIOD = 10
_C.TRAIN.STAGE2.USE_CHECKPOINT = False


# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

# ---------------------------------------------------------------------------- #
# DEVICE
# ---------------------------------------------------------------------------- #
_C.DEVICE = CN()
_C.DEVICE.NAME = 'cuda'
# If has multi GPUs, options: 'True', 'False'
_C.DEVICE.DIST = True
_C.DEVICE.LOCAL_RANK = 0

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = "./outputs"
