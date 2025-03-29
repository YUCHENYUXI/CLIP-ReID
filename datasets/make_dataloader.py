import torch
import torchvision.transforms as T
import datasets.transforms.spatial_transforms as ST
import datasets.transforms.temporal_transforms as TT
from torch.utils.data import DataLoader

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from datasets.viddatasets.samplers import RandomIdentitySampler as RandomIdentitySampler_vid
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .occ_duke import OCC_DukeMTMCreID
from .vehicleid import VehicleID
from .veri import VeRi
from datasets.viddatasets.video_loader import VideoDataset

from datasets.viddatasets.data_manager import Mars
import datasets.viddatasets.data_manager as data_manager
__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,
    'MARS': Mars
}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def make_dataloader(cfg):
    isVID=cfg.DATASETS.ISVID
    dataset_name = cfg.DATASETS.NAMES
    
    cfg_num_instances = cfg.DATALOADER.NUM_INSTANCE
    cfg_num_workers=cfg.DATALOADER.NUM_WORKERS
    batch_size = cfg.SOLVER.IMS_PER_BATCH
    test_batch_size = cfg.TEST.IMS_PER_BATCH    
    #%% 定义变换器
    pin_memory = True if cfg.DATALOADER.PIN else False
    num_workers = cfg.DATALOADER.NUM_WORKERS

    if not isVID:
        #%% 处理图像ReID数据集
        # 图像变换器
        train_transforms = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.Pad(cfg.INPUT.PADDING),
                T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
                RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
                # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
            ])

        val_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])
        
        # 重点！获取数据集 # 1. 通过cfg.DATASETS.NAMES获取数据集的名称，然后通过__factory获取数据集的类
        dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
        train_set = ImageDataset(dataset.train, train_transforms)
        train_set_normal = ImageDataset(dataset.train, val_transforms)
        num_classes = dataset.num_train_pids
        cam_num = dataset.num_train_cams
        view_num = dataset.num_train_vids

        # 接下来是构建DataLoader，这里有两个DataLoader，一个是训练集的DataLoader，一个是验证集的DataLoader
        if 'triplet' in cfg.DATALOADER.SAMPLER:
            if cfg.MODEL.DIST_TRAIN:
                print('DIST_TRAIN START')
                mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
                data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
                batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
                train_loader = torch.utils.data.DataLoader(
                    train_set,
                    num_workers=num_workers,
                    batch_sampler=batch_sampler,
                    collate_fn=train_collate_fn,
                    pin_memory=True,
                )
            else:
                train_loader = DataLoader(
                    train_set, 
                    batch_size=cfg.SOLVER.IMS_PER_BATCH,
                    sampler=RandomIdentitySampler(
                        dataset.train,
                        cfg.SOLVER.IMS_PER_BATCH, 
                        cfg.DATALOADER.NUM_INSTANCE
                    ),
                    num_workers=num_workers, 
                    collate_fn=train_collate_fn
                )
        elif cfg.DATALOADER.SAMPLER == 'softmax':
            print('using softmax sampler')
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
                collate_fn=train_collate_fn
            )
        else:
            print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

        val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

        val_loader = DataLoader(
            val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn
        )
        train_loader_normal = DataLoader(
            train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn
        )
        return train_loader, val_loader, len(dataset.query), num_classes, cam_num, view_num
    else:
        #%% 视频数据集
        spatial_transform_train = ST.Compose([
                    ST.Scale(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                    ST.RandomHorizontalFlip(),
                    ST.ToTensor(),
                    ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        temporal_transform_train = TT.TemporalRandomCrop(size=cfg.DATALOADER.SEQLEN, stride=cfg.DATALOADER.SAMSTREDE)

        spatial_transform_test = ST.Compose([
                    ST.Scale(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                    ST.ToTensor(),
                    ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        temporal_transform_test = TT.TemporalBeginCrop()
        #%% 数据集
        print("Initializing dataset {}".format(dataset_name))
        dataset = data_manager.init_dataset(name=dataset_name, root=cfg.DATASETS.ROOT_DIR)

        #%% dataloader
        if cfg.DATASETS.NAMES != 'MARS':
            trainloader = DataLoader(
                VideoDataset(dataset.train_dense, spatial_transform=spatial_transform_train, temporal_transform=temporal_transform_train),
                sampler=RandomIdentitySampler_vid(dataset.train_dense, num_instances=cfg_num_instances),
                batch_size=batch_size, num_workers=cfg_num_workers,
                pin_memory=pin_memory, drop_last=True)
        else:
            trainloader = DataLoader(
            VideoDataset(dataset.train, spatial_transform=spatial_transform_train, temporal_transform=temporal_transform_train),
            sampler=RandomIdentitySampler_vid(dataset.train, num_instances=cfg_num_instances),
            batch_size=batch_size, num_workers=cfg_num_workers,
            pin_memory=pin_memory, drop_last=True)

        queryloader = DataLoader(
            VideoDataset(dataset.query, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test),
            batch_size=test_batch_size, shuffle=False, num_workers=0,
            pin_memory=pin_memory, drop_last=False)

        galleryloader = DataLoader(
            VideoDataset(dataset.gallery, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test),
            batch_size=test_batch_size, shuffle=False, num_workers=0,
            pin_memory=pin_memory, drop_last=False)
        # return train_loader, val_loader, len(dataset.query), num_classes, cam_num, view_num
        view_num = None
        num_classes = dataset.num_train_pids
        cam_num = dataset.num_cameras
        view_num = dataset.num_view
        return trainloader, val_loader, len(dataset.query), num_classes, cam_num, view_num

