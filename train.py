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
    train_loader = build_dataloader(cfg, True)
    test_loader = build_dataloader(cfg, False)

    trainer = Trainer(cfg, logger)


    # Training Stage One
    model = build_training_model(cfg)
    if cfg.DEVICE.NAME == "cuda":
        model.cuda()        # move to GPU before dist
    if cfg.DEVICE.DIST:
        model = DDP(model, device_ids=[cfg.LOCAL_RANK])

    criterion_1 = build_criterion(cfg, stage=1)
    optimizer_1 = build_optimizer(cfg.TRAIN.STAGE1, model)
    scheduler_1 = build_scheduler(cfg.TRAIN.STAGE1, optimizer_1, len(train_loader))

    trainer.train_one('pair', model, train_loader, test_loader, criterion_1, optimizer_1, scheduler_1, cfg.TRAIN.STAGE1)

    # Training Stage Two
    model = build_cls_model(cfg, model.encoder)
    if cfg.DEVICE.DIST:
        model = DDP(model, device_ids=[cfg.LOCAL_RANK])

    criterion_2 = build_criterion(cfg, stage=2)
    optimizer_2 = build_partial_optimizer(model, ['classifier'], ['encoder'], cfg.TRAIN.STAGE2.OPTIMIZER.NAME, cfg.TRAIN.STAGE2.OPTIMIZER.PARAMS)
    scheduler_2 = build_scheduler(cfg.TRAIN.STAGE2, optimizer_2, len(train_loader)) # check n_iters

    trainer.train_two('classification', model, train_loader, test_loader, criterion_2, optimizer_2, scheduler_2, cfg.TRAIN.STAGE2)

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
    parser.add_argument("opts", help="Modify config options from list, e.g.: 'DIST' True", 
                        default=None, nargs=argparse.REMAINDER)
    # parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.TRAIN.SEED)

    # Initialize Dist
    if cfg.DEVICE.DIST:
        from torch.nn.parallel import DistributedDataParallel as DDP
        cfg.defrost()
        cfg.DEVICE.LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        cfg.freeze()
        torch.cuda.set_device(cfg.LOCAL_RANK)   # args.local_rank
        # if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")

        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        torch.distributed.barrier()

    # Setting Output Directory
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = create_logger(os.path.join(output_dir, 'logs'), disk_rank=rank, name=args.name)
    logger.info(args.info)
    # logger.info(args)
    # if args.config_file != "":
    #     logger.info("Loaded configuration file {}".format(args.config_file))
    #     with open(args.config_file, 'r') as cf:
    #         config_str = "\n" + cf.read()
    #         logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    
    main(cfg, logger)