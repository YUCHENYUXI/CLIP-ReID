import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import time
from datetime import timedelta
import math
def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    device = "cuda"

    logger = logging.getLogger("RAR.train")
    logger.info('Start training')
    
    model.to(local_rank)
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs for training'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    IDloss_meter = AverageMeter()
    TRILoss_meter = AverageMeter()
    R1_mAP_evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    GradScaler = amp.GradScaler()

    train_start_t = time.monotonic()
    logger.info("Model: {}".format(model))

    # 模型
    if cfg.MODEL.IS_USE_CKPT:
        ckpt_path_resume = os.path.normpath(cfg.MODEL.CKPT_PATH)
        model.load_state_dict(torch.load(ckpt_path_resume))
        logger.info(f"Load checkpoint from {ckpt_path_resume}")
        epoch = cfg.MODEL.LAST_OR_NEW_CKPT_EPOCH_MARK

    # 周期
    for epoch in range(1, cfg.SOLVER.MAX_EPOCHS + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        R1_mAP_evaluator.reset()
        IDloss_meter.reset()
        TRILoss_meter.reset()

        model.train()
        test_mode = not cfg.MODEL.TRAIN_MODE
        n_iter = 0
        # 批次
        if cfg.MODEL.TRAIN_MODE:
            for n_iter, dpac in enumerate(train_loader):
                # dpac {'aer': aer,'rgb': rgb,'pid': pid,'cid': camid}
                optimizer.zero_grad()
                optimizer_center.zero_grad()
                # 数据
                if torch.cuda.is_available():
                    dpac['rgb'] = dpac['rgb'].to(device,non_blocking=True)
                    dpac['aer'] = dpac['aer'].to(device,non_blocking=True)
                    dpac['pid'] = dpac['pid'].to(device,non_blocking=True)
                    if cfg.MODEL.SIE_CAMERA:
                        dpac['cid'] = dpac['cid'].to(device,non_blocking=True)
                # 前向
                with torch.cuda.amp.autocast():
                    score, feat = model(dpac)
                    # 反向
                    loss, idloss, triloss = loss_fn(score, feat, dpac['pid'], dpac['cid'])

                    if not math.isfinite(loss.item()):
                        logger.warning(f"Non-finite loss at epoch {epoch}, iter {n_iter}. Skipping.")
                        continue

                GradScaler.scale(loss).backward()
                GradScaler.step(optimizer)
                GradScaler.update()

                if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                    for param in center_criterion.parameters():
                        if param.grad is not None:
                            param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                    GradScaler.step(optimizer_center)
                    GradScaler.update()

                loss_meter.update(loss.item(), dpac['rgb'].shape[0])
                acc_meter.update(acc.item(), 1)
                IDloss_meter.update(idloss, 1)
                TRILoss_meter.update(triloss, 1)

                if (n_iter + 1) % cfg.SOLVER.LOG_PERIOD == 0:
                    logger.info(f"Epoch[{epoch}] Iter[{n_iter+1}/{len(train_loader)}] "
                                f"AVGLoss: {loss_meter.avg:.3f}, AVGID: {IDloss_meter.avg:.3f}, AVGTri: {TRILoss_meter.avg:.3f}, "
                                f"AVGAcc: {acc_meter.avg:.3f}, LR: {scheduler.get_lr()[0]:.2e}")

        scheduler.step()

        end_time = time.time()
        time_per_batch = (end_time - start_time) / max(1, n_iter + 1)
        logger.info(f"Epoch {epoch} done. Time per batch: {time_per_batch:.3f}s, "f"Speed: {train_loader.batch_size / time_per_batch:.1f} samples/s")


        if (epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0) or (epoch in cfg.SOLVER.STEPS):
            if cfg.MODEL.IS_USE_CKPT:
                ckpt_path = os.path.join(cfg.OUTPUT_DIR, f"Base_{ckpt_path_resume}_New_{cfg.MODEL.NAME}_Plus{epoch}.pth")
            else:
                ckpt_path = os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.NAME}_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

        if test_mode:
            ckpt_path = r""
            # 加载模型
            model.load_state_dict(torch.load(ckpt_path))
            logger.info(f"TEST--Load checkpoint from {ckpt_path}")

        # if test_mode or epoch % cfg.SOLVER.EVAL_PERIOD == 0:
        #     try:
        #         model.eval()
        #         print("Testing")
        #         for n_iter, dpac in enumerate(val_loader):
        #             rgb = dpac['rgb'].to(device,non_blocking=True)
        #             aer = dpac['aer'].to(device,non_blocking=True)

        #             cids= dpac['cid'].tolist()
        #             pid=dpac['pid']

        #             cid_t = torch.tensor(cids).to(device,non_blocking=True) if cfg.MODEL.SIE_CAMERA else None
        #             with torch.no_grad():
                        
        #                 feat = model(video, cam_label=cam_id, view_label=target_view)
        #                 feat = feat.view(batch_size, num_frames, -1).mean(dim=1)
        #                 R1_mAP_evaluator.update((feat, pids, cids))
                    
        #         cmc, mAP, *_ = R1_mAP_evaluator.compute()
        #         logger.info(f"Validation Results - Epoch: {epoch}")
        #         logger.info(f"mAP: {mAP:.1%}")
        #         for r in [1, 5, 10]:
        #             logger.info(f"CMC curve, Rank-{r:<3}: {cmc[r-1]:.1%}")
        #     except Exception as e:
        #         logger.exception(f"Evaluation failed at epoch {epoch}: {e}")
            # torch.cuda.empty_cache()
        
        
        if test_mode:
            break

    total_time = timedelta(seconds=time.monotonic() - train_start_t)
    logger.info(f"Total running time: {total_time}")
    print(cfg.OUTPUT_DIR)


