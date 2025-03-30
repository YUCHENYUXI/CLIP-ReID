from utils.logger import setup_logger
from datasets.make_dataloader import make_dataloader
from model.make_model import make_model
from solver.make_optimizer import make_optimizer
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from processor.processor import do_train,do_train_vid
import random
import torch
import numpy as np
import os
import argparse
from config import cfg_base as cfg
print("Current working directory:", os.getcwd())
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
def add_args(parser:argparse.ArgumentParser):
    parser.add_argument(
        "--config_file", default="configs/person/vit_base.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    return

if __name__ == '__main__':
    # cmd cfg
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    add_args(parser)
    args = parser.parse_args()

    # file cfg
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    # set seed
    set_seed(cfg.SOLVER.SEED)
    # distributed?set dev
    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)
    # logger
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)
    # show cfg
    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    # distributed? init grp
    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # update cuda env configs
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID 
    train_loader, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    # model  ;loss_function  ;optimizer and scheduler
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                  cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
    # do train
    if cfg.DATASETS.ISVID:# 视频ReID
        logger.info("Training with videos ReID")
        do_train_vid(
            cfg,
            model,
            center_criterion,
            train_loader,
            val_loader,
            optimizer,
            optimizer_center,
            scheduler,
            loss_func,
            num_query, args.local_rank
        )
    else:# 图像ReID
        logger.info("Training with images ReID")
        do_train(
            cfg,
            model,
            center_criterion,
            train_loader,#
            val_loader,#
            optimizer,
            optimizer_center,
            scheduler,
            loss_func,
            num_query, args.local_rank
        )
