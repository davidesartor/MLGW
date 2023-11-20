import torch


def conv_block(idx: int, hidden_size=32, kernel_size=3):
    return torch.nn.Sequential(
        torch.nn.LazyConv1d(hidden_size, kernel_size, dilation=2**idx, padding="same"),  # type: ignore
        torch.nn.LazyBatchNorm1d(),
        torch.nn.ReLU(),
    )


def linear_block(idx: int, hidden_size=32):
    return torch.nn.Sequential(
        torch.nn.LazyLinear(out_features=hidden_size),
        torch.nn.LazyBatchNorm1d(),
        torch.nn.ReLU(),
    )
