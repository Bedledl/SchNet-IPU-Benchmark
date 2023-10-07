"""
This file contains test models for each operation
"""
import math

import torch.nn.functional

"""
backward not supported

index add
abs
softplus
linalg norm?
norm?
"""
from math import log

from torch import nn, Tensor, functional, norm, linalg, index_select, sqrt


# replaced ops

class IndexAddModel(nn.Module):
    def forward(self, target: Tensor, src: Tensor, index: Tensor):
        return target.index_add(0, index, src)

class AbsModel(nn.Module):
    def forward(self, x: Tensor):
        return x.abs()

class SoftPlusModel(nn.Module):
    def forward(self, x: Tensor):
        return functional.softplus(x) - log(2.0)


class NormModel(nn.Module):
    def forward(self, x: Tensor):
        return norm(x, dim=1)


class LinalgNormModel(nn.Module):
    def forward(self, x: Tensor):
        return linalg.norm(x, dim=1)


class PowSumSqrtModel(nn.Module):
    def forward(self, x: Tensor):
        return sqrt(x.pow(2).sum(-1))


class IndexModel(nn.Module):
    def forward(self, x: Tensor, index: Tensor):
        return x[index]


class IndexSelectModel(nn.Module):
    def forward(self, x: Tensor, index: Tensor):
        return index_select(x, 0, index)


class GatherModel(nn.Module):
    """
    This model expects 2-dimensional input at the x
    """
    def forward(self, x: Tensor, index: Tensor):
        index_expanded = index.unsqueeze(1).expand(index.shape[0], x.size(1))
        return x.gather(0, index_expanded)


class TorchSoftplusModel(nn.Module):
    def forward(self, x: Tensor):
        out =  torch.nn.functional.softplus(x).to(torch.float32)
        return torch.round(out, decimals=2)


class SoftplusLogExpModel(nn.Module):
    def forward(self, x: Tensor):
        out = torch.log(1 + torch.exp(x))
        return torch.round(out, decimals=2)


class WhereNotInPlaceModel(nn.Module):
    def forward(self, x: Tensor, cutoff: int):
        x_gt_cutoff = (x < cutoff).float()
        x = x * x_gt_cutoff
        return x


class WhereInPlaceModel(nn.Module):
    def forward(self, x: Tensor, cutoff: int):
        x *= (x < cutoff).float()
        return x


class CosineCutoffCPUModel(nn.Module):
    def forward(self, x: Tensor, cutoff: int):
        input_cut = 0.5 * (torch.cos(x * math.pi / cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        input_cut *= (x < cutoff).float()
        return input_cut

class CosineCutoffIPUModel(nn.Module):
    def forward(self, x: Tensor, cutoff: int):
        input_cut = 0.5 * (torch.cos(x * math.pi / cutoff) + 1.0)
        input_cut = input_cut * (x < cutoff).float()
        return input_cut

class TopKDistance(nn.Module):
    def forward(self, positions: Tensor, k: int, ):
        i_expanded = positions.expand(positions.size(0), *positions.shape)
        j_expanded = positions.reshape(positions.size(0), 1, positions.size(1))

        diff = i_expanded - j_expanded

        norm = torch.linalg.norm(diff, dim=-1)

        dist, col = torch.topk(norm,
                               k=k + 1,  # we need k + 1 because topk includes loops
                               dim=-1,
                               largest=False)
        # somehow when using this distance values the gradients after the filter network are zero
        # but they are the same values as we get with the Distance transform

        # this removes all the loops
        col = col.reshape(-1, k + 1)[:, 1:].reshape(-1)

        return dist


#replacement ops
class ScatterAddModel(nn.Module):
    def forward(self, target: Tensor, src: Tensor, index: Tensor):
        return target.scatter_add(0, index, src)