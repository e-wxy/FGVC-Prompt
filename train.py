from utils.logger import create_logger
from data import build_dataloader
from model import build_training_model, build_cls_model
from solver import build_scheduler, build_optimizer, Trainer, build_partial_optimizer
from loss import build_criterion
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
from torch.nn.parallel import DistributedDataParallel as DDP

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main(cfg, logger):
    # Data
    logger.info("Building Dataset")
    train_loader = build_dataloader(cfg, True)
    test_loader = build_dataloader(cfg, False)

    if cfg.DEVICE.DIST:
        batch_per_epoch = (len(train_loader) + int(os.environ['WORLD_SIZE']) - 1) // int(os.environ['WORLD_SIZE'])
    else:
        batch_per_epoch = len(train_loader)

    trainer = Trainer(cfg, logger)


    # Training Stage One
    model = build_training_model(cfg)
    if cfg.DEVICE.NAME == "cuda":
        model.cuda()    # Move to GPU before dist
    if cfg.DEVICE.DIST:
        model = DDP(model, device_ids=[cfg.DEVICE.LOCAL_RANK])

    criterion_1 = build_criterion(cfg, stage=1)
    optimizer_1 = build_optimizer(cfg.TRAIN.STAGE1, model)
    scheduler_1 = build_scheduler(cfg.TRAIN.STAGE1, optimizer_1, batch_per_epoch)

    trainer.train_one('pair', model, train_loader, test_loader, criterion_1, optimizer_1, scheduler_1, cfg.TRAIN.STAGE1)

    # Training Stage Two
    if cfg.DEVICE.DIST:
        model = model.module
    model = build_cls_model(cfg, model.encoder)
    if cfg.DEVICE.NAME == "cuda":
        model.cuda()
    if cfg.DEVICE.DIST:
        model = DDP(model, device_ids=[cfg.DEVICE.LOCAL_RANK])

    criterion_2 = build_criterion(cfg, stage=2)
    optimizer_2 = build_partial_optimizer(model, ['classifier'], ['encoder'], cfg.TRAIN.STAGE2.OPTIMIZER.NAME, cfg.TRAIN.STAGE2.OPTIMIZER.PARAMS)
    scheduler_2 = build_scheduler(cfg.TRAIN.STAGE2, optimizer_2, batch_per_epoch) 

    trainer.train_two('cls', model, train_loader, test_loader, criterion_2, optimizer_2, scheduler_2, cfg.TRAIN.STAGE2)

    trainer.record_training_process()




if __name__ == '__main__':

    # Argparse & Config
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('-c', "--cfg_file", default="", 
                        help="Path to config file", type=str)
    parser.add_argument('-n', "--name", default="test1", 
                        help="Name of the logger", type=str)
    parser.add_argument('-i', "--info", default="", 
                        help="Info about this run", type=str)
    parser.add_argument("opts", help="Modify config options from list, e.g.: DEVICE.DIST True", 
                        default=None, nargs=argparse.REMAINDER)
    # parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.cfg_file != "":
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.TRAIN.SEED)

    # Initialize Dist
    if cfg.DEVICE.DIST:
        cfg.defrost()
        cfg.DEVICE.LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        cfg.freeze()
        torch.cuda.set_device(cfg.DEVICE.LOCAL_RANK)   # args.local_rank
        # if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")

        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        torch.distributed.barrier()
    else:
        rank = 0

    # Setting Output Directory
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        if rank == 0:
            os.makedirs(output_dir)

    logger = create_logger(os.path.join(output_dir, 'logs'), dist_rank=rank, name=args.name)
    logger.info(args.info)
    # logger.info(args)
    # if args.config_file != "":
    #     logger.info("Loaded configuration file {}".format(args.config_file))
    #     with open(args.config_file, 'r') as cf:
    #         config_str = "\n" + cf.read()
    #         logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    
    main(cfg, logger)