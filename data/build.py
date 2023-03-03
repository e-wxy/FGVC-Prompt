import os
from torchvision import transforms
from timm.data import create_transform
from timm.data.transforms import str_to_pil_interp
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.distributed as dist
from torch.utils import data

from .cub_text import CUBDataset, RandomPermutateDrop

def build_dataloader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.DATA.DATALOADER.BATCH_SIZE
    else:
        batch_size = cfg.DATA.DATALOADER.TEST_BATCH_SIZE

    dataset = build_dataset(cfg, is_train)
    if cfg.DEVICE.DIST:
        sampler = data.DistributedSampler(
            dataset, 
            num_replicas=dist.get_world_size(), 
            rank=dist.get_rank(), 
            shuffle=True
        )
        data_loader = data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=cfg.DATA.DATALOADER.NUM_WORKERS,
            pin_memory=cfg.DATA.DATALOADER.PIN_MEMORY,
            drop_last=True,
        )
    else:
        data_loader = data.DataLoader(
            dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=cfg.DATA.DATALOADER.NUM_WORKERS,
            pin_memory=cfg.DATA.DATALOADER.PIN_MEMORY,
            drop_last=True,
        )
    return data_loader


def build_dataset(cfg, is_train=True):
    transform = build_transform(cfg, is_train)
    root = cfg.DATA.DATASET.ROOT_DIR
    if cfg.DATA.DATASET.NAME == 'cub':
        dataset = CUBDataset(os.path.join(root, 'cub2002011'), transform, RandomPermutateDrop(cfg.DATA.DATASET.DROP_RATE), is_train)
    return dataset

def build_transform(cfg, is_train=True):
    resize_im = cfg.DATA.DATALOADER.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=cfg.DATA.DATALOADER.IMG_SIZE,
            is_training=True,
            color_jitter=cfg.DATA.AUG.COLOR_JITTER if cfg.DATA.AUG.COLOR_JITTER > 0 else None,
            auto_augment=cfg.DATA.AUG.AUTO_AUGMENT if cfg.DATA.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=cfg.DATA.AUG.REPROB,
            re_mode=cfg.DATA.AUG.REMODE,
            re_count=cfg.DATA.AUG.RECOUNT,
            interpolation=cfg.DATA.DATALOADER.TRAIN_INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                cfg.DATA.DATALOADER.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if cfg.DATA.DATALOADER.TEST_CROP:
            size = int((256 / 224) * cfg.DATA.DATALOADER.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=str_to_pil_interp(
                    cfg.DATA.DATALOADER.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(cfg.DATA.DATALOADER.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((cfg.DATA.DATALOADER.IMG_SIZE, cfg.DATA.DATALOADER.IMG_SIZE),
                                  interpolation=str_to_pil_interp(cfg.DATA.DATALOADER.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
