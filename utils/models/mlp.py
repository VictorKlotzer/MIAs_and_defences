import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['mlp1']

class MLP(nn.Module):
    """
    Simple MLP model 
    """
    def __init__(self, nb_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3*32*32, nb_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x

def mlp1(nb_classes):
    return MLP(nb_classes)


def test():
    net = mlp1(nb_classes = 10)
    y = net(torch.randn(42, 3, 32, 32))
    print(y.size())

if __name__ == "__main__":
    test()