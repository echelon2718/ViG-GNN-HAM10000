import math
import torch
from torch import nn
import torch.nn.functional as F

def pairwise_distance(x):
    with torch.no_grad():
        x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim = -1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)

def part_pairwise_distance(x, start_idx=0, end_idx=1):
    """
    Menghitung jarak euclidean antar titik dalam graf.
    Argumen:
        - x: tensor 3 dimensi (batch, jumlah_titik, jumlah_dimensi)
    
    Mengembalikan:
        - jarak euclidean: (batch, jumlah_titik, jumlah_titik)
    """

    with torch.no_grad():
        x_part = x[:, start_idx:end_idx]
        x_square_part = torch.sum(torch.mul(x_part, x_part), dim=-1, keepdim=True)
        x_inner = -2 * torch.matmul(x_part, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square_part + x_inner + x_square.transpose(2, 1)

def xy_pairwise_distance(x, y):
    """
    Menghitung jarak pasangan x,y dalam tensor
    Argumen:
        x: tensor(batch_size, jumlah_titik, jumlah_dimensi)
    Mengembalikan:
        pairwise_distance : tensor(batch_size, jumlah_titik, jumlah_titik)
    """
    with torch.no_grad():
        xy_inner = -2*torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1)

def dense_knn_matrix(x, k=16, relative_pos=None):
    """
    Mendapatkan jarak antar titik terdekat berbasis KNN
    Argumen:
        - x : tensor 4 dimensi (batch, jumlah_dimensi, jumlah_titik, 1)
        - k : jumlah tetangga terdekat yang ingin diambil (int)
    Mengembalikan:
        - nearest neighbor matrix : tensor (batch, jumlah_titik, k), tensor (batch, jumlah_titik, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        n_part = 10000
        if n_points > n_part:
            nn_idx_list = []
            groups = math.ceil(n_points / n_part)
            for i in range(groups):
                start_idx = n_part * i
                end_idx = min(n_points, n_part * (i + 1))
                dist = part_pairwise_distance(x.detach(), start_idx, end_idx)
                if relative_pos is not None:
                    dist += relative_pos[:, start_idx:end_idx]
                _, nn_idx_part = torch.topk(-dist, k=k)
                nn_idx_list += [nn_idx_part]
            nn_idx = torch.cat(nn_idx_list, dim=1)
        else:
            dist = pairwise_distance(x.detach())
            if relative_pos is not None:
                dist += relative_pos
            _, nn_idx = torch.topk(-dist, k=k)
        
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)

    return torch.stack((nn_idx, center_idx), dim=0)

def xy_dense_knn_matrix(x, y, k=16, relative_pos=None):
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = xy_pairwise_distance(x.detach(), y.detach())
        if relative_pos is not None:
            dist += relative_pos
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)

class DenseDilated(nn.Module):
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
    
    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        
        return edge_index

class DenseDilatedKnnGraph(nn.Module):
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)
    
    def forward(self, x, y=None, relative_pos=None):
        if y is not None:
            x = F.normalize(x, p=2.0, dim=1)
            y = F.normalize(y, p=2.0, dim=1)
            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation, relative_pos)
        
        else:
            x = F.normalize(x, p=2.0, dim=1)
            edge_index = dense_knn_matrix(x, self.k * self.dilation, relative_pos)
        
        return self._dilated(edge_index)