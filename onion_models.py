import torch.nn as nn
import torch

class OnionEncoder(nn.Module):
    # input_image: (batch_size, height, width, channel)
    # hidden_size: (batch_size, height, width, channel)
    # 
    def __init__(self):
        super(OnionEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 1, 5)
        )
    def forward(self, x):
        # x : [batch_size, height, width, channel]
        x = self.encoder(x)
        return x

class OnionDecoder(nn.Module):
    def __init__(self):
        super(OnionDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 8, 5),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=3, padding=1),
            nn.Tanh(),
            nn.InstanceNorm2d(1, 1, 100, 124)
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

class OnionNet(nn.Module):
    def __init__(self):
        super(OnionNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(225, 200),
            nn.ReLU(True),
            nn.Linear(200, 200),
            nn.ReLU(True),
            nn.Linear(200, 221),
            nn.ReLU(True)
            
        )

    def forward(self, x):
        x = self.model(x)
        return x
