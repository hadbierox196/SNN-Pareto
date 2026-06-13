"""
ANN Baseline Model — snn-pareto
Week 1: Shared backbone, ReLU activations.
This same backbone is converted to SNN in Week 2.
"""

import torch
import torch.nn as nn


class ANN_Baseline(nn.Module):
    """
    Shared convolutional backbone for Fashion-MNIST.

    Architecture:
        Conv(1->32, 3x3) -> ReLU -> MaxPool(2x2)
        Conv(32->64, 3x3) -> ReLU -> MaxPool(2x2)
        Flatten -> Linear(3136->256) -> ReLU -> Dropout(0.3)
        Linear(256->10)

    Parameters: 824,458 (all trainable)
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ANN_Baseline()
    print(model)
    print(f"Trainable parameters: {count_parameters(model):,}")
