

# def do_inference(cfg, model, val_loader, num_query):
#     import numpy as np
#     device = "cuda"
#     logger = logging.getLogger("RGBE.test")
#     logger.info("Enter inferencing")

#     R1_mAP_evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
#     R1_mAP_evaluator.reset()

#     if device:
#         if torch.cuda.device_count() > 1:
#             print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
#             model = nn.DataParallel(model)
#         model.to(device,non_blocking=True)

#     model.eval()
#     valid_samples = 0

#     for n_iter, datapac in enumerate(val_loader):
#         try:
#             camids = torch.tensor(camid, device=device).clone().detach()
#             target_view = None

#             with torch.no_grad():
#                 imgs = imgs.to(device,non_blocking=True)
#                 if cfg.MODEL.SIE_CAMERA:
#                     camids = camids.to(device,non_blocking=True)
#                 else: 
#                     camids = None

#                 if cfg.MODEL.SIE_VIEW:
#                     target_view = target_view.to(device,non_blocking=True)
#                 else: 
#                     target_view = None

#                 batch_size, channels, num_frames, height, width = imgs.shape

#                 all_feats = []
#                 for i in range(num_frames):
#                     frame = imgs[:, :, i, :, :]
#                     feat_i = model(frame, cam_label=camids, view_label=target_view)
                    
#                     # 安全检查
#                     if feat_i is None or isinstance(feat_i, (list, tuple)) and any(f is None for f in feat_i):
#                         logger.warning(f"Inference error: model returned None at frame {i}, skipping sample.")
#                         raise ValueError("Invalid feature output")

#                     all_feats.append(feat_i)

#                 # 融合 features
#                 feat = torch.stack([all_feats[t] for t in range(num_frames)]).mean(dim=0)

#                 # 特征有效性检查
#                 if torch.isnan(feat).any() or torch.isinf(feat).any():
#                     logger.warning(f"Inference warning: NaN or Inf in features for sample {n_iter}, skipping.")
#                     continue
#                 if torch.norm(feat, p=2, dim=1).mean() < 1e-6:
#                     logger.warning(f"Inference warning: feature norm is too small at sample {n_iter}, skipping.")
#                     continue

#                 R1_mAP_evaluator.update((feat, vid, camid))
#                 valid_samples += 1

#         except Exception as e:
#             logger.error(f"Inference failed at sample {n_iter}: {e}")
#             continue

#     if valid_samples == 0:
#         logger.error("No valid features were extracted from inference. Please check model or input format.")
#         empty_cmc = np.zeros(50, dtype=np.float32)
#         return empty_cmc[0], empty_cmc[4]

#     try:
#         cmc, mAP, _, _, _, _, _ = R1_mAP_evaluator.compute()
#         logger.info("Validation Results ")
#         logger.info("mAP: {:.1%}".format(mAP))
#         for r in [1, 5, 10]:
#             logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
#         return cmc[0], cmc[4]
#     except Exception as e:
#         logger.error(f"R1_mAP_evaluator failed to compute metrics: {e}")
#         empty_cmc = np.zeros(50, dtype=np.float32)
#         return empty_cmc[0], empty_cmc[4]
