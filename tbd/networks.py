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


class ConvNet1d(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Sequence[int],
        out_channels: int,
        kernel_size=3,
        batch_norm=True,
        activation: torch.nn.Module = DEFAULT_ACTIVATION,
        out_activation: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        channels = [in_channels] + list(hidden_channels) + [out_channels]
        activations = [activation for _ in channels[1:-1]] + [out_activation]
        for i, (in_ch, out_ch, act) in enumerate(zip(channels[:-1], channels[1:], activations)):
            stride = 1 + kernel_size // 2 if i % 2 else 1
            padding = kernel_size // 2 if i % 2 else "same"
            self.append(torch.nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding))
            if batch_norm:
                self.append(torch.nn.BatchNorm1d(out_ch))
            if act is not None:
                self.append(act)


class ConvNet2d(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Sequence[int],
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        batch_norm=True,
        activation: torch.nn.Module = DEFAULT_ACTIVATION,
        out_activation: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        channels = [in_channels] + list(hidden_channels) + [out_channels]
        activations = [activation for _ in channels[1:-1]] + [out_activation]
        for i, (in_ch, out_ch, act) in enumerate(zip(channels[:-1], channels[1:], activations)):
            stride = (1 + kernel_size[0] // 2, 1 + kernel_size[1] // 2) if i % 2 else 1
            padding = (kernel_size[0] // 2, kernel_size[1] // 2) if i % 2 else "same"
            self.append(torch.nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding))
            if batch_norm:
                self.append(torch.nn.BatchNorm2d(out_ch))
            if act is not None:
                self.append(act)
