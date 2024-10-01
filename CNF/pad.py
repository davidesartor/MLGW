class DownSample(nn.Module):
    def __init__(self, channels, down=4, contract=2):
        super().__init__()
        self.proj = nn.Linear(channels * down, channels * down // contract)
        self.down, self.contract = down, contract

    def forward(self, x):
        *batch, lenght, channels = x.shape
        x = x.reshape(*batch, lenght // self.down, -1)
        x = self.proj(x)
        return x


class UpSample(nn.Module):
    def __init__(self, channels, up=4, expand=2):
        super().__init__()
        self.proj = nn.Linear(channels, channels * expand)
        self.up, self.expand = up, expand

    def forward(self, x):
        *batch, lenght, channels = x.shape
        x = self.proj(x)
        x = x.reshape(*batch, lenght * self.up, -1)
        return x


class ConvResBlock(nn.Module):
    def __init__(self, channels, activation=nn.GELU()):
        super().__init__()
        self.activation = activation
        self.respath = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, 5, padding="same"),
            activation,
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, 5, padding="same"),
        )

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.activation(x + self.respath(x))
        return x.transpose(-1, -2)
