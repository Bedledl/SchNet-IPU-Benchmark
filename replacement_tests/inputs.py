from torch import tensor, ones, arange, float32

# target, src, index
scatter_add_input = [
    ones(10, dtype=float32),
    arange(5, dtype=float32),
    tensor([0, 0, 2, 4, 6])
]

# src, index
index_select_input = [
    tensor([
        [6, 4, 2, 5],
        [4, 6, 7, 1],
        [8, 3, 1, 4]
    ], dtype=float32),
    tensor([0, 0, 2])
]

index_select_weight_mult = [
tensor([
        [6, 4, 2, 5],
        [4, 6, 7, 1],
        [8, 3, 1, 4]
    ], dtype=float32),
    tensor([0, 0, 2]),
    tensor([1, 0, 5], dtype=float32)
]

# 3d tensors
norm_input = [
    tensor([
        [6, 4, 2, 5],
        [4, 6, 7, 1],
        [8, 3, 1, 4]
    ], dtype=float32)
]

topk_input = [
    tensor([
        [0, 0, 0],
        [0, 2, 0],
        [5, 7, 8],
        [5, 7, 9]
    ], dtype=float32),
    2
]

softplus_input = [
    tensor([0, 5, 2, -1, -9], dtype=float32)
]

cutoff_input = [
    tensor([0, 0.3, 0.5, 0.75, 1, 1.2], dtype=float32),
    0.5
]
