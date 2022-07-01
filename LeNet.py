from audioop import avg
import torch
from torch import avg_pool1d, nn
from d2l import torch as d2l

class Reshape(torch.nn.Module):
    def __init__(self, x):
        super().__init__()
    def forward(self, x):
        return x.view(-1, 1, 28, 28)

net = torch.nn.Sequential(
    #Reshape(),
    nn.Conv2d(1, 6, kernel_size=5, padding=2),  #28-5+4+1=28
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16*5*5, 120),
    nn.Sigmoid(),
    nn.Linear(120, 84), 
    nn.Sigmoid(),
    nn.Linear(84, 10)
)

net2 = torch.nn.Sequential(
    #Reshape(),
    nn.Conv2d(1, 6, kernel_size=5, padding=2),  #28-5+4+1=28
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16*5*5, 120),
    nn.ReLU(),
    nn.Linear(120, 84), 
    nn.ReLU(),
    nn.Linear(84, 10)
)
def get_shape():
    #查看各层的输出形状
    X=torch.rand(size=(1,1,28,28), dtype=torch.float32)
    for layer in net:
        X=layer(X)
        print(layer.__class__.__name__, 'shape:\t', X.shape)
