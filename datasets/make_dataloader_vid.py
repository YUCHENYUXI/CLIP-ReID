import torch

def make_dataloader(cfg):
    import datasets.vid_utils.transforms.spatial_transforms as ST 
    import datasets.vid_utils.transforms.temporal_transforms as TT

    # Data augmentation
    spatial_transform_train = ST.Compose([
                ST.Scale(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                ST.RandomHorizontalFlip(),
                ST.ToTensor(),
                ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    temporal_transform_train = TT.TemporalRandomCrop(\
        size=cfg.INPUT.seq_len, stride=cfg.INPUT.sample_stride)

    spatial_transform_test = ST.Compose([
                ST.Scale(cfg.INPUT.SIZE_TEST, interpolation=3),
                ST.ToTensor(),
                ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    temporal_transform_test = TT.TemporalBeginCrop()
    
    import  datasets.vid_utils.tools.data_manager as data_manager
    
    dataset = data_manager.init_dataset(name=cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    pin_memory = True if torch.cuda.is_available() else False


    from torch.utils.data import DataLoader

    from .vid_utils.tools.video_loader import VideoDataset
    from .vid_utils.tools.samplers import RandomIdentitySampler

    if cfg.DATASETS.NAMES != 'mars':
        trainloader = DataLoader(
            VideoDataset(dataset.train_dense, \
                    spatial_transform=spatial_transform_train, \
                        temporal_transform=temporal_transform_train),
            sampler=RandomIdentitySampler(dataset.train_dense, num_instances=cfg.DATALOADER.NUM_INSTANCE),
            batch_size=cfg.SOLVER.IMS_PER_BATCH, num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=pin_memory, drop_last=True)
    else:
        trainloader = DataLoader(
            VideoDataset(dataset.train, spatial_transform=spatial_transform_train, temporal_transform=temporal_transform_train),
            sampler=RandomIdentitySampler(dataset.train, num_instances=cfg.DATALOADER.NUM_INSTANCE),
            batch_size=cfg.SOLVER.IMS_PER_BATCH, num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=pin_memory, drop_last=True)

    val_loader = DataLoader(
        VideoDataset(dataset.query + dataset.gallery, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test),
        batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=0,
        pin_memory=pin_memory, drop_last=True)
    
    train_loader = trainloader
    cam_num = dataset.cam_num
    view_num = dataset.view_num
    num_classes = dataset.num_train_pids
    
    return train_loader, val_loader, len(dataset.query), num_classes, cam_num, view_num
