import torch
NL='\n'



def make_optimizer(cfg, model, center_criterion):
    logtxt = ""
    logtxt += "create-opt----------------------------------------------------mizer"+NL
    # --- freeze ---
    for name, param in model.named_parameters():
        if param.requires_grad is False:
            logtxt+=(f"ign: {name}"+NL)
            continue
        if "clip.visual." in name or "clip.transformer." in name:
            # 排除 text_projection, logit_scale, positional_embedding, token_embedding
            # 这些通常希望在微调中更新，除非有特殊策略
            if "clip.text_projection" in name or \
               "clip.logit_scale" in name or \
               "clip.positional_embedding" in name or \
               "clip.token_embedding" in name:
                param.requires_grad = True # 确保这些参数是可训练的
            else:
                param.requires_grad = False # 冻结图像和文本Transformer的主干层
                logtxt+=(f"fre: {name}"+NL)

        else:
            param.requires_grad = True # 其他层（如classifier, bottleneck）默认可训练
            


    params = []
    for key, value in model.named_parameters():
        if value.requires_grad is False:
            continue
        
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key: # bias - lr * cfg.SOLVER.BIAS_LR_FACTOR
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR: # fc - lr * cfg.SOLVER.BIAS_LR_FACTOR
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        logtxt+=(f"opt: {key}"+NL)

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)

    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)
    return optimizer, optimizer_center, logtxt