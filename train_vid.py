from utils.logger import setup_logger
from datasets.make_dataloader_vid import make_dataloader
from model.make_model import make_model
from solver.make_optimizer import make_optimizer
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from processor.processor_vid import do_train
import random
import torch
import numpy as np
import os
import argparse
from config.defaults_base import _C as cfg
def init_all():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--config_file", required=True, help="path to config file", type=str)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    #
    if cfg.OUTPUT_DIR and not os.path.exists(cfg.OUTPUT_DIR): 
        os.makedirs(cfg.OUTPUT_DIR)
    logger = setup_logger("RAR", cfg.OUTPUT_DIR, if_train=True) # RGB-AER-PAR_text-ReID
    logger.info("SavingPath:{}".format(cfg.OUTPUT_DIR))
    logger.info(rf"CLI_args: {args}")
    # if args.config_file != "":
    #     logger.info("Configuration file:{}".format(args.config_file))
    #     with open(args.config_file, 'r') as cf:
    #         config_str = cf.read()
    #         logger.info("Configuration file content:\n{}".format(config_str))
    logger.info("Running with config:\n{}".format(cfg))
    logger.info("END_INIT\n")
    return cfg,args,logger
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 更保险
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 
if __name__ == '__main__':
    print('*'*30,"PID = %d, GPU ID = %d" % (os.getpid(), torch.cuda.current_device()),'*'*30)
    cfg ,args,logger= init_all()
    # set_global_seed(cfg.SOLVER.SEED) # debug 不用，更快
    dataloaders = make_dataloader(cfg)
    model = make_model(cfg, num_class=dataloaders['cls_num'], camera_num=dataloaders['cam_num'], view_num = dataloaders['view_num'])
    loss_func, center_criterion,log_l = make_loss(cfg, num_classes=dataloaders['cls_num'])
    optimizer, optimizer_center, log_o = make_optimizer(cfg, model, center_criterion)
    logger.info("Loss:{}".format(log_l))
    logger.info("Optimizer:{}".format(log_o))
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    logger.info(f"模型加载后显存使用情况,当前分配显存: {torch.cuda.memory_allocated(args.local_rank) / (1024**3):.2f} GB")
    do_train(
        cfg,
        model,
        center_criterion,
        dataloaders['train'],
        dataloaders['val'],
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        dataloaders['query_num'],
        args.local_rank
    )