import torch
from torch import nn

class CenterLossSimplified(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLossSimplified, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        # 将 centers 放在相应的设备上
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)

        # === 简化点 1: 计算欧氏距离的平方 ===
        # distmat[i][j] = ||x[i] - centers[j]||^2
        # PyTorch 提供了 torch.cdist 来计算距离，但它计算的是 L2 距离。
        # 如果要计算 L2 距离的平方，使用 ||a - b||^2 = ||a||^2 + ||b||^2 - 2ab.T 这种展开式仍然是最常见的，
        # 因为它在数值上更稳定，并且可以利用优化的矩阵乘法。
        # 所以 distmat 的计算保持不变，因为它已经很高效了。
        
        # 计算 x 的平方和
        # x_norm_sq = torch.sum(x**2, dim=1, keepdim=True) # (batch_size, 1)
        # 计算 centers 的平方和
        # centers_norm_sq = torch.sum(self.centers**2, dim=1, keepdim=True) # (num_classes, 1)

        # 展开计算欧氏距离的平方矩阵
        # (x_norm_sq (batch_size, 1) + centers_norm_sq.t() (1, num_classes))
        # - 2 * x @ self.centers.t() (batch_size, num_classes)
        distmat = torch.sum(x**2, dim=1, keepdim=True)  - torch.matmul(x, self.centers.t()) + torch.sum(self.centers**2, dim=1, keepdim=True).t()  - torch.matmul(x, self.centers.t()) 
        # distmat = x_norm_sq + centers_norm_sq.t() - 2 * torch.matmul(x, self.centers.t())
        distmat = distmat.clamp(min=1e-12, max=1e+12) 
        # === 简化点 2: 使用高级索引直接提取距离 ===
        # labels 是 (batch_size,)，表示每个样本对应的类别索引
        # distmat 是 (batch_size, num_classes)
        # 我们可以直接使用 labels 作为索引来从 distmat 中选择每个样本对应的距离
        # torch.arange(batch_size) 产生 0 到 batch_size-1 的索引
        # labels 提供了每个样本对应的列索引
        
        # dist_to_centers 是一个 (batch_size,) 维的 Tensor，包含了每个样本到其对应类中心的欧氏距离的平方
        dist_to_centers = distmat[torch.arange(batch_size), labels.long()]

        # === 简化点 3: 数值稳定性 (可选，如果计算过程中出现负值，通常是浮点误差) ===
        # 理论上欧氏距离的平方不应该是负数。如果出现极小的负数，clamp可以防止log等操作报错。
        # 也可以考虑使用 F.relu(dist_to_centers) 或 torch.abs(dist_to_centers)
        # dist_to_centers = dist_to_centers.clamp(min=1e-12, max=1e+12) 

        # 计算平均损失
        loss = dist_to_centers.mean()

        return loss

class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat = torch.addmm(distmat, x, self.centers.t(),alpha=-2)  # distmat[i][j] = ||x[i] - centers[j]||^2

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: 
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()
        return loss


if __name__ == '__main__':
    seed = 0
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 更保险
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 

    use_gpu = True
    center_loss = CenterLoss(use_gpu=use_gpu)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 更保险
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 
    center_loss_S = CenterLossSimplified(use_gpu=use_gpu)
    features = torch.rand(16, 2048).cuda()
    targets = torch.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).cuda()

    loss = center_loss(features, targets)
    loss_S = center_loss_S(features, targets)
    print(loss,loss_S)


