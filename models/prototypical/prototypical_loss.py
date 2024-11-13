# File: prototypical_loss.py
# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module

class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function.
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors.
    '''
    return torch.cdist(x, y, p=2)  # Simplified to use PyTorch's cdist

def prototypical_loss(input, target, n_support):
    '''
    Compute prototypical loss and accuracy.
    '''
    device = input.device
    classes = torch.unique(target)
    n_classes = len(classes)
    support_idxs = [target.eq(c).nonzero(as_tuple=True)[0][:n_support] for c in classes]
    query_idxs = [target.eq(c).nonzero(as_tuple=True)[0][n_support:] for c in classes]

    prototypes = torch.stack([input[idx].mean(0) for idx in support_idxs])
    query_samples = torch.cat([input[idx] for idx in query_idxs], dim=0)
    query_labels = torch.cat([torch.full((len(idx),), c, dtype=torch.long, device=device)
                              for c, idx in enumerate(query_idxs)])

    dists = euclidean_dist(query_samples, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1)

    loss_val = F.nll_loss(log_p_y, query_labels)
    _, y_hat = log_p_y.max(1)
    acc_val = (y_hat == query_labels).float().mean()

    return loss_val, acc_val
