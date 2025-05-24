import torch
import torch.nn as nn
import torch.nn.functional as F

def hard_example_mining(dist_mat: torch.Tensor, 
                        labels: torch.LongTensor, 
                        return_inds: bool = False):
    """For each anchor, find the hardest positive and negative sample."""
    N = dist_mat.size(0)
    # mask for positives and negatives
    labels = labels.view(N, 1)
    is_pos = labels.eq(labels.t())
    is_neg = labels.ne(labels.t())

    # hardest positive: max over positives
    dist_ap, relative_p_inds = dist_mat.masked_fill(~is_pos, float('-inf')).max(dim=1)
    # hardest negative: min over negatives
    dist_an, relative_n_inds = dist_mat.masked_fill(~is_neg, float('inf')).min(dim=1)

    if return_inds:
        return dist_ap, dist_an, relative_p_inds, relative_n_inds
    return dist_ap, dist_an


class TripletLoss(nn.Module):
    """Triplet loss with hard example mining and optional feature normalization."""

    def __init__(self, margin: float = None, hard_factor: float = 0.0):
        super().__init__()
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def forward(self,
                features: torch.Tensor,
                labels: torch.LongTensor,
                normalize_features: bool = False) -> torch.Tensor:
        # optional L2-normalize features
        if normalize_features:
            features = F.normalize(features, p=2, dim=-1)

        # pairwise distance matrix
        dist_mat = torch.cdist(features, features, p=2)

        # hard mining
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        # apply hard factor scaling
        dist_ap = dist_ap * (1.0 + self.hard_factor)
        dist_an = dist_an * (1.0 - self.hard_factor)

        # prepare target
        target = torch.ones_like(dist_an)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, target)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, target)
        return loss, dist_ap, dist_an

# test case
if __name__ == "__main__":
    features = torch.randn(10, 128)  # 10 samples, 128-dimensional features
    labels = torch.randint(0, 5, (10,))  # 5 classes
    triplet_loss = TripletLoss(margin=0.2, hard_factor=0.1)
    loss, dist_ap, dist_an = triplet_loss(features, labels)
    print(f"Loss: {loss.item()}, Dist_ap: {dist_ap.mean().item()}, Dist_an: {dist_an.mean().item()}")