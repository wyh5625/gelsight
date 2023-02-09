from torch import nn


class MLPEncoder(nn.Module):
    def __init__(self, encoded_space_dim = 2):
        super().__init__()

        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )

    def forward(self, x):

        x = self.encoder_lin(x)
        return x

class MLPEncoder_NOXY(nn.Module):
    def __init__(self, encoded_space_dim = 2):
        super().__init__()

        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )

    def forward(self, x):

        x = self.encoder_lin(x)
        return x