import datasets.vid_utils.transforms.spatial_transforms as ST 
import datasets.vid_utils.transforms.temporal_transforms as TT
from torch.utils.data import DataLoader
from .vid_utils.tools.video_loader import AERDataset
from .vid_utils.tools.samplers import RandomIdentitySampler
import  datasets.vid_utils.tools.data_manager as data_manager

def make_dataloader(cfg):
    """
    returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        num_query: number of query samples
        num_classes: number of classes
        camera_num: number of cameras
        view_num: number of views
    """
    # Data augmentation
    # train
    spatial_transform_train =ST.Compose([ # 训练集-空间
        ST.Scale(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        ST.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        ST.RandomHorizontalFlip(),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    temporal_transform_train =TT.TemporalCenterStrideCrop( # 训练集-时间
        size=cfg.INPUT.seq_len,
        stride=cfg.INPUT.sample_stride
    )
    # test
    spatial_transform_test = ST.Compose([
        ST.Scale(cfg.INPUT.SIZE_TEST, interpolation=3),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    temporal_transform_test = TT.TemporalCenterStrideCrop(
        size=cfg.INPUT.seq_len,
        stride=cfg.INPUT.sample_stride
    )

    # dataset
    dataset=data_manager.init_dataset(name=cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    cam_num=len(set(dataset.cam_ids))
    is_pin = True
    res ={"train":DataLoader(
                    AERDataset(
                        dataset.train, 
                        spatial_transform=spatial_transform_train, #
                        temporal_transform=temporal_transform_train#
                        ),
                    sampler=RandomIdentitySampler(
                        dataset.train, 
                        num_instances=cfg.DATALOADER.NUM_INSTANCE
                        ),
                    batch_size=cfg.SOLVER.VIDS_PER_BATCH, 
                    num_workers=cfg.DATALOADER.NUM_WORKERS,
                    pin_memory=is_pin, 
                    drop_last=True,
                    persistent_workers=True,
                ),
          "val":DataLoader(
                AERDataset(dataset.query + dataset.gallery, 
                            spatial_transform=spatial_transform_test, 
                            temporal_transform=temporal_transform_test
                            ),
                batch_size=cfg.TEST.VIDS_PER_BATCH, 
                shuffle=False, 
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                pin_memory=is_pin, 
                drop_last=True,
                persistent_workers=True,
            ),
          "query_num":len(dataset.query),
          "cls_num": dataset.num_train_pids,
          "cam_num": cam_num,
          "view_num": None
          }
    return res