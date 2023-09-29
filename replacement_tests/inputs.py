from torch import tensor, ones, arange, float32

# target, src, index
scatter_add_input = [
    ones(10, dtype=float32),
    arange(5, dtype=float32),
    tensor([0, 0, 2, 4, 6])
]

index_select_input = [
    tensor([
        [6, 4, 2, 5],
        [4, 6, 7, 1],
        [8, 3, 1, 4]
    ], dtype=float32),
    tensor([0, 0, 2])
]
