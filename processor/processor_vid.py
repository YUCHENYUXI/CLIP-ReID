import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch import amp

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
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("RGB-E.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    IDlossmeter=AverageMeter()
    TRILossmeter=AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        IDlossmeter.reset()
        TRILossmeter.reset()
        


        model.train()

        # for n_iter, (vids, pids, target_cam) in enumerate(train_loader):
        n_iter = 0
        if 0==1:
            vids, pids, target_cam = None
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            vids = vids.to(device)
            target = pids.to(device)
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else: 
                target_cam = None
            # if cfg.MODEL.SIE_VIEW:
            #     target_view = target_view.to(device)
            # else: 
            target_view = None

            #---
            batch_size, num_frames, channels, height, width = vids.shape  # (32,4,3,256,128)
            
            vids=vids.view([-1,channels,height,width]) # (128,3,256,128)
            target_train=torch.stack([target for i in range(num_frames)]).view(num_frames,batch_size)
            target_train=target_train.permute(1,0).reshape(-1)
            with torch.amp.autocast('cuda',enabled=True):  # 使用混合精度加速
                score, feat = model(vids, target_train, cam_label=target_cam, view_label=target_view)
                # print(f"iter: {n_iter}")
                for i in range(len(score)):
                    score[i]=score[i].view(batch_size,num_frames,-1).mean(dim=1)
                for i in range(len(feat)):
                    feat[i]=feat[i].view(batch_size,num_frames,-1).mean(dim=1)

                
                # 计算损失
                loss,idloss,triloss = loss_fn(score, feat, target, target_cam)
            #---

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), vids.shape[0])
            acc_meter.update(acc, 1)
            IDlossmeter.update(idloss, 1)
            TRILossmeter.update(triloss, 1)

            torch.cuda.synchronize()
            if ((n_iter + 1) % log_period) == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] AVG - Loss: {:.3f}, ID:{:.3f}, Tri:{:.3f}, Acc: {:.3f} | Base Lr: {:.2e} | VAL - Loss: {:.3f}, ID:{:.3f}, Tri:{:.3f}, Acc: {:.3f}".format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg,IDlossmeter.avg,TRILossmeter.avg, acc_meter.avg, scheduler.get_lr()[0],loss_meter.val,IDlossmeter.val,TRILossmeter.val, acc_meter.val))
 
        scheduler.step()

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if (epoch % checkpoint_period ==0) or ((epoch) in cfg.SOLVER.STEPS ):
            torch.save(model.state_dict(),
                        os.path.join(cfg.OUTPUT_DIR, 
                                     cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0 :
            model.eval()
            for n_iter, (video, target_id, cam_id) in enumerate(val_loader): # 
                video = video.to(device)
                if cfg.MODEL.SIE_CAMERA:
                    cam_id = cam_id.to(device)
                else:
                    cam_id = None
                target_view = None
                with torch.no_grad():
                    #---
                    batch_size, num_frames, channels, height, width = video.shape  # (32,3,4,256,128)
                    video=video.view([-1,channels,height,width]) # (128,3,256,128)
                    feat = model(video, cam_label=cam_id, view_label=target_view)
                    #---
                    feat=feat.view(batch_size,num_frames,-1).mean(dim=1)
                    if cam_id is None:
                        cam_id = [0*batch_size]
                    evaluator.update((feat, target_id, cam_id))
                    
                    # 计算损失
            
            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)


    
def do_inference(cfg, model, val_loader, num_query):
    import numpy as np
    device = "cuda"
    logger = logging.getLogger("RGBE.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    valid_samples = 0

    for n_iter, (imgs, vid, camid) in enumerate(val_loader):
        try:
            camids = torch.tensor(camid, device=device).clone().detach()
            target_view = None

            with torch.no_grad():
                imgs = imgs.to(device)
                if cfg.MODEL.SIE_CAMERA:
                    camids = camids.to(device)
                else: 
                    camids = None

                if cfg.MODEL.SIE_VIEW:
                    target_view = target_view.to(device)
                else: 
                    target_view = None

                batch_size, channels, num_frames, height, width = imgs.shape

                all_feats = []
                for i in range(num_frames):
                    frame = imgs[:, :, i, :, :]
                    feat_i = model(frame, cam_label=camids, view_label=target_view)
                    
                    # 安全检查
                    if feat_i is None or isinstance(feat_i, (list, tuple)) and any(f is None for f in feat_i):
                        logger.warning(f"Inference error: model returned None at frame {i}, skipping sample.")
                        raise ValueError("Invalid feature output")

                    all_feats.append(feat_i)

                # 融合 features
                feat = torch.stack([all_feats[t] for t in range(num_frames)]).mean(dim=0)

                # 特征有效性检查
                if torch.isnan(feat).any() or torch.isinf(feat).any():
                    logger.warning(f"Inference warning: NaN or Inf in features for sample {n_iter}, skipping.")
                    continue
                if torch.norm(feat, p=2, dim=1).mean() < 1e-6:
                    logger.warning(f"Inference warning: feature norm is too small at sample {n_iter}, skipping.")
                    continue

                evaluator.update((feat, vid, camid))
                valid_samples += 1

        except Exception as e:
            logger.error(f"Inference failed at sample {n_iter}: {e}")
            continue

    if valid_samples == 0:
        logger.error("No valid features were extracted from inference. Please check model or input format.")
        empty_cmc = np.zeros(50, dtype=np.float32)
        return empty_cmc[0], empty_cmc[4]

    try:
        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        logger.info("Validation Results ")
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        return cmc[0], cmc[4]
    except Exception as e:
        logger.error(f"Evaluator failed to compute metrics: {e}")
        empty_cmc = np.zeros(50, dtype=np.float32)
        return empty_cmc[0], empty_cmc[4]
