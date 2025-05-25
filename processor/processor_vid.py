import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import time
from datetime import timedelta
def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, 
             local_rank):
    # loss_func, center_criterion
    # optimizer, optimizer_center
    device = "cuda"
    logger = logging.getLogger("RAR.train")
    logger.info('Start training')
    ## dist(No implementation)
    model.to(local_rank)
    if torch.cuda.device_count() > 1:
        logger.info('Using %d GPUs for training', torch.cuda.device_count())
        model = nn.DataParallel(model)
    ## Metric Trackers 
    loss_meter, acc_meter, IDloss_meter, TRILoss_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    R1_mAP_evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    GradScaler = amp.GradScaler()
    ## timer
    train_start_t = time.monotonic()
    logger.info("Model: %s", model)
    logger.info("%s", "-="*20+"-")

    # 模型
    if cfg.MODEL.IS_USE_CKPT:
        ckpt_path_resume = os.path.normpath(cfg.MODEL.CKPT_PATH)
        model.load_state_dict(torch.load(ckpt_path_resume))
        logger.info("Load checkpoint from %s", ckpt_path_resume)
        epoch = cfg.MODEL.LAST_OR_NEW_CKPT_EPOCH_MARK

    def reset_me_tras():
        loss_meter.reset()
        acc_meter.reset()
        R1_mAP_evaluator.reset()
        IDloss_meter.reset()
        TRILoss_meter.reset()
    # epoch loop
    for epoch in range(1, cfg.SOLVER.MAX_EPOCHS + 1):
        start_time = time.time()
        reset_me_tras()
        model.train()
        test_mode = not cfg.MODEL.TRAIN_MODE
        idx_bat = 0
        # batch loop
        if cfg.MODEL.TRAIN_MODE:
            for idx_bat, dpac in enumerate(train_loader):
                logger.info("data loaded, now VRAM: %.2f GB", torch.cuda.memory_allocated(device) / (1024**3))
                # dpac {'aer': aer,'rgb': rgb,'pid': pid,'cid': camid}
                optimizer.zero_grad()
                optimizer_center.zero_grad()
                # data to device
                if torch.cuda.is_available():
                    dpac['rgb'], dpac['aer'], dpac['pid'] = dpac['rgb'].to(device,non_blocking=True), dpac['aer'].to(device,non_blocking=True), dpac['pid'].to(device,non_blocking=True)
                    if cfg.MODEL.SIE_CAMERA:
                        dpac['cid'] = dpac['cid'].to(device,non_blocking=True)
                # train
                with torch.cuda.amp.autocast():
                    score, feat = model(dpac)
                    #
                    loss, idloss, triloss = loss_fn(score, feat, dpac['pid'], dpac['cid']) # ?cen?
                    # 
                    R1_mAP_evaluator.update((feat, pids, cids))
                    
  
                GradScaler.scale(loss).backward()
                GradScaler.step(optimizer)
                GradScaler.update()

                if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                    for param in center_criterion.parameters():
                        param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                        scaler.step(optimizer_center)
                        scaler.update()
                if isinstance(score, list):
                    acc = (score[0].max(1)[1] == target).float().mean()
                else:
                    acc = (score.max(1)[1] == target).float().mean()

                # cal metrics
                acc_meter.update(acc, 1)
                loss_meter.update(loss.item(), dpac['rgb'].shape[0])
                IDloss_meter.update(idloss, 1)
                TRILoss_meter.update(triloss, 1)
                if (idx_bat + 1) % cfg.SOLVER.LOG_PERIOD == 0:
                    logger.info(
                        "Epo%d Bat%d/%d AVG-Loss: %.3f, -ID: %.3f, -Tri: %.3f, -Acc: %.3f; NOW-Loss: %.3f, -ID: %.3f, -Tri: %.3f, -Acc: %.3f, LR: %.2e",
                        epoch, idx_bat+1, len(train_loader), 
                        loss_meter.avg, IDloss_meter.avg, TRILoss_meter.avg, acc_meter.avg,
                        loss.item(), idloss, triloss, acc, scheduler.get_lr()[0]
                    )
                del dpac, loss, idloss, triloss, score, feat
                torch.cuda.empty_cache() # 清空 PyTorch 的缓存分配器中未使用的内存
        # batch end
        logger.info("峰值显存: %.2f GB", torch.cuda.max_memory_allocated(device) / (1024**3))
        scheduler.step() # warmupscheduler
        ###################
        cmc, mAP, *_ = R1_mAP_evaluator.compute()
        logger.info("Trainset Results - Epoch: %d", epoch)
        logger.info("mAP: %.1f%%", mAP * 100)
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-%-3d: %.1f%%", r, cmc[r-1] * 100)
        R1_mAP_evaluator.reset()  # reset for next epoch
        ###one epoch end###
        end_time = time.time()
        time_per_batch = (end_time - start_time) / max(1, idx_bat + 1)
        logger.info("Ep %d. %.3fs/batch, %.1f tracks/s", epoch, time_per_batch, train_loader.batch_size / time_per_batch)
        ###################
        if (epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0) or (epoch in cfg.SOLVER.STEPS):
            if cfg.MODEL.IS_USE_CKPT:
                ckpt_path = os.path.join(cfg.OUTPUT_DIR, f"Base_{ckpt_path_resume}_New_{cfg.MODEL.NAME}_Plus{epoch}.pth")
            else:
                ckpt_path = os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.NAME}_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            logger.info("Saved checkpoint to %s", ckpt_path)

        ### TEST ###
        if test_mode:
            ckpt_path = r""
            # 加载模型
            model.load_state_dict(torch.load(ckpt_path))
            logger.info("TEST-Load checkpoint from %s", ckpt_path)
        # if test_mode or epoch % cfg.SOLVER.EVAL_PERIOD == 0:
        #     try:
        #         model.eval()
        #         logger.info("Testing")
        #         for idx_bat, dpac in enumerate(val_loader):
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
            # torch.cuda.empty_cache()
        
        
        if test_mode:
            break

    total_time = timedelta(seconds=time.monotonic() - train_start_t)
    logger.info("Total running time: %s", total_time)
    logger.info("%s", cfg.OUTPUT_DIR)


