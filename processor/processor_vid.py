import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch import amp
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

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("RGB-E.train")
    logger.info('Start training')
    
    model.to(local_rank)
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs for training'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    IDlossmeter = AverageMeter()
    TRILossmeter = AverageMeter()
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()

    all_start_time = time.monotonic()
    logger.info("Model: {}".format(model))

    if cfg.MODEL.RESUME:
        ckpt_path_resume = os.path.normpath(cfg.MODEL.CHECKPOINT)
        # 加载模型
        model.load_state_dict(torch.load(ckpt_path_resume))
        logger.info(f"Load checkpoint from {ckpt_path_resume}")
        resume_epoch = cfg.MODEL.CHECKPOINT_EPOCH
        epoch = resume_epoch


    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        IDlossmeter.reset()
        TRILossmeter.reset()

        model.train()
        train_mode = cfg.MODEL.TRAIN_MODE
        test_mode = not train_mode
        n_iter = 0
        if train_mode:
            for n_iter, (vids, pids, target_cam) in enumerate(train_loader):
                try:
                    optimizer.zero_grad()
                    optimizer_center.zero_grad()

                    vids = vids.to(device)
                    target = pids.to(device)
                    target_cam = target_cam.to(device) if cfg.MODEL.SIE_CAMERA else None
                    target_view = None

                    batch_size, num_frames, channels, height, width = vids.shape
                    vids = vids.view(-1, channels, height, width)
                    target_train = torch.stack([target for _ in range(num_frames)]).view(num_frames, batch_size)
                    target_train = target_train.permute(1, 0).reshape(-1)

                    with torch.amp.autocast('cuda', enabled=True):
                        score, feat = model(vids, target_train, cam_label=target_cam, view_label=target_view)

                        # check for None or invalid output
                        if score is None or feat is None:
                            logger.warning(f"Model output is None at epoch {epoch}, iter {n_iter}")
                            continue
                        if isinstance(score, list):
                            for s in score:
                                if not torch.isfinite(s).all():
                                    logger.warning("Non-finite score detected, skipping this batch.")
                                    continue
                        if isinstance(feat, list):
                            for f in feat:
                                if not torch.isfinite(f).all():
                                    logger.warning("Non-finite feature detected, skipping this batch.")
                                    continue

                        for i in range(len(score)):
                            score[i] = score[i].view(batch_size, num_frames, -1).mean(dim=1)
                        for i in range(len(feat)):
                            feat[i] = feat[i].view(batch_size, num_frames, -1).mean(dim=1)

                        loss, idloss, triloss = loss_fn(score, feat, target, target_cam)

                        if not math.isfinite(loss.item()):
                            logger.warning(f"Non-finite loss at epoch {epoch}, iter {n_iter}. Skipping.")
                            continue

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                        for param in center_criterion.parameters():
                            if param.grad is not None:
                                param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                        scaler.step(optimizer_center)
                        scaler.update()

                    if isinstance(score, list):
                        acc = (score[0].max(1)[1] == target).float().mean()
                    else:
                        acc = (score.max(1)[1] == target).float().mean()

                    loss_meter.update(loss.item(), vids.shape[0])
                    acc_meter.update(acc.item(), 1)
                    IDlossmeter.update(idloss, 1)
                    TRILossmeter.update(triloss, 1)

                    if (n_iter + 1) % log_period == 0:
                        logger.info(f"Epoch[{epoch}] Iter[{n_iter+1}/{len(train_loader)}] "
                                    f"AVGLoss: {loss_meter.avg:.3f}, AVGID: {IDlossmeter.avg:.3f}, AVGTri: {TRILossmeter.avg:.3f}, "
                                    f"AVGAcc: {acc_meter.avg:.3f}, LR: {scheduler.get_lr()[0]:.2e}")

                except Exception as e:
                    logger.exception(f"Exception during training at epoch {epoch}, iter {n_iter}: {e}")
                    continue  # 防止因模型未训练或偶发错误中断整个流程

        scheduler.step()

        end_time = time.time()
        time_per_batch = (end_time - start_time) / max(1, n_iter + 1)
        logger.info(f"Epoch {epoch} done. Time per batch: {time_per_batch:.3f}s, "
                    f"Speed: {train_loader.batch_size / time_per_batch:.1f} samples/s")


        if (epoch % checkpoint_period == 0) or (epoch in cfg.SOLVER.STEPS):
            if cfg.MODEL.RESUME:
                ckpt_path = os.path.join(cfg.OUTPUT_DIR, f"Base_{ckpt_path_resume}_New_{cfg.MODEL.NAME}_Plus{epoch}.pth")
            else:
                ckpt_path = os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.NAME}_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

        if test_mode:
            ckpt_path = r"res/vit_rgb_lrD/ViT-B-16_100.pth"
            # 加载模型
            model.load_state_dict(torch.load(ckpt_path))
            logger.info(f"TEST--Load checkpoint from {ckpt_path}")

        if test_mode or epoch % eval_period == 0:
            try:
                model.eval()
                print("Testing")
                for n_iter, (video, target_id, cam_id) in enumerate(val_loader):
                    video = video.to(device)
                    cams= cam_id.tolist()
                    cam_id = cam_id.to(device) if cfg.MODEL.SIE_CAMERA else None
                    target_view = None
                    with torch.no_grad():
                        batch_size, num_frames, channels, height, width = video.shape
                        video = video.view(-1, channels, height, width)
                        feat = model(video, cam_label=cam_id, view_label=target_view)
                        feat = feat.view(batch_size, num_frames, -1).mean(dim=1)
                        evaluator.update((feat, target_id, cams))
                    
                cmc, mAP, *_ = evaluator.compute()
                logger.info(f"Validation Results - Epoch: {epoch}")
                logger.info(f"mAP: {mAP:.1%}")
                for r in [1, 5, 10]:
                    logger.info(f"CMC curve, Rank-{r:<3}: {cmc[r-1]:.1%}")
            except Exception as e:
                logger.exception(f"Evaluation failed at epoch {epoch}: {e}")
            torch.cuda.empty_cache()
        
        
        if test_mode:
            break

    total_time = timedelta(seconds=time.monotonic() - all_start_time)
    logger.info(f"Total running time: {total_time}")
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
