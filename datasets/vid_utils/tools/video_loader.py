from logging import warning
import os
import torch
import torch.utils.data as data
from PIL import Image

def video_loader(img_paths):
    video = []
    for image_path in img_paths:
        if os.path.exists(image_path):
            video.append(Image.open(image_path))
        else:
            warning(f"Image path {image_path} does not exist.")
    return video

class VideoDataset(data.Dataset):
    """Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """
    def __init__(self, dataset, spatial_transform=None, temporal_transform=None,):
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = video_loader
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (clip, pid, camid) where pid is identity of the clip.
        """
        img_paths, pid, camid = self.dataset[index]

        if self.temporal_transform is not None:
            img_paths = self.temporal_transform(img_paths)

        clip = self.loader(img_paths)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        # trans T x C x H x W to C x T x H x W
        clip = torch.stack(clip, 0) # .permute(1, 0, 2, 3)

        return clip, pid, camid

class AERDataset(data.Dataset):
    """RGB-AER Person ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """
    def __init__(self, dataset, spatial_transform=None, temporal_transform=None,):
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = video_loader
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
        dict——
            'aer': aer,
            'rgb': rgb,
            'pid': pid,
            'camid': camid
        """
        img_paths, pid, camid = self.dataset[index]
        if self.temporal_transform is not None:
            img_paths = self.temporal_transform(img_paths)
        # load video # replace
        aer_path=[]
        rgb_path=[]
        aer = self.loader(img_paths)
        rgb = self.loader(img_paths)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            aer = [self.spatial_transform(img) for img in aer]
            rgb = [self.spatial_transform(img) for img in rgb]
        # trans T x C x H x W to C x T x H x W
        aer = torch.stack(aer, 0)
        rgb = torch.stack(rgb, 0)
        # return aer, rgb, pid, camid
        return {
            'aer': aer,
            'rgb': rgb,
            'pid': pid,
            'camid': camid
        }