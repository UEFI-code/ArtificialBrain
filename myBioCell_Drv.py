import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import myBionicCell_cuda

class myBioCellFunction(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, weight, remain, cellenergymax):
        output = myBionicCell_cuda.forward(input, weight, remain, cellenergymax)
        return output[0]

class myBioCell(nn.Module):
    def __init__(self, input_features, output_features, cellenergymax):
        super(myBioCell, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.weight.data.uniform_(-0.1, 0.1)
        self.energymax = cellenergymax
        self.remain = torch.ones(output_features) * cellenergymax

    def forward(self, input):
        return myBioCellFunction.apply(input, self.weight, self.remain, self.energymax)
