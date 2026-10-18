"""Red neuronal convolucional (CNN) del trabajo 1."""
import torch.nn as nn


class CNN(nn.Module):
    """3 bloques [conv-BN-ReLU x2 + MaxPool] y una capa final.

    Usa AdaptiveAvgPool, así acepta imágenes de cualquier tamaño (32x32 en
    CIFAR-10, 64x64 en datasets propios).
    """

    def __init__(self, n_clases: int, canales=(32, 64, 128)):
        super().__init__()
        capas, c_in = [], 3
        for c in canales:
            capas += [
                nn.Conv2d(c_in, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c_in = c
        self.features = nn.Sequential(*capas)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(c_in, n_clases),
        )

    def forward(self, x):
        return self.head(self.features(x))
