"""
This file contains test models for each operation
"""

"""
backward not supported

index add
abs
softplus
linalg norm?
norm?
"""
from math import log

from torch import nn, Tensor, functional, norm, linalg, index_select


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
        return norm(x)


class LinalgNormModel(nn.Module):
    def forward(self, x: Tensor):
        return norm(x)


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

#replacement ops

class ScatterAddModel(nn.Module):
    def forward(self, target: Tensor, src: Tensor, index: Tensor):
        return target.scatter_add(0, index, src)