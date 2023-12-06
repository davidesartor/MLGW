from typing import Sequence, Optional
import torch

DEFAULT_ACTIVATION = torch.nn.LeakyReLU()


class LinearNet(torch.nn.Sequential):
    def __init__(
        self,
        in_features: int,
        hidden_features: Sequence[int],
        out_features: int,
        batch_norm=True,
        activation: torch.nn.Module = DEFAULT_ACTIVATION,
        out_activation: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        features = [in_features] + list(hidden_features) + [out_features]
        activations = [activation for _ in features[1:-1]] + [out_activation]
        for in_f, out_f, act in zip(features[:-1], features[1:], activations):
            self.append(torch.nn.Linear(in_f, out_f))
            if batch_norm:
                self.append(torch.nn.BatchNorm1d(out_f))
            if act is not None:
                self.append(act)


class ConvNet(torch.nn.Sequential):
    def __init__(
        self,
        conv_dim: int,
        in_channels: int,
        hidden_channels: Sequence[int],
        out_channels: int,
        kernel_size=3,
        batch_norm=True,
        activation: torch.nn.Module = DEFAULT_ACTIVATION,
        out_activation: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        if conv_dim == 1:
            conv_cls = torch.nn.Conv1d
            bn_cls = torch.nn.BatchNorm1d
        elif conv_dim == 2:
            conv_cls = torch.nn.Conv2d
            bn_cls = torch.nn.BatchNorm2d
        elif conv_dim == 3:
            conv_cls = torch.nn.Conv3d
            bn_cls = torch.nn.BatchNorm3d
        else:
            raise ValueError(f"conv_dim must be 1, 2, or 3, not {conv_dim}")

        channels = [in_channels] + list(hidden_channels) + [out_channels]
        activations = [activation for _ in channels[1:-1]] + [out_activation]
        for i, (in_ch, out_ch, act) in enumerate(zip(channels[:-1], channels[1:], activations)):
            self.append(conv_cls(in_ch, out_ch, kernel_size, dilation=2**i, padding="same"))
            if batch_norm:
                self.append(bn_cls(out_ch))
            if act is not None:
                self.append(act)
