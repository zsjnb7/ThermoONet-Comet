
import torch
import torch.nn as nn
import numpy as np
import math

"""Following is the architecture of ThermoONet"""
# Note that it is necessary to run the code in GPU for rapid computation
# This requires to install the PyTorch that matches with your own computer's GPU
# The step is simple to follow the method of PyTorch official

# Construct a channel attention mechanism module (more sensitive to features)
class SELayer(nn.Module):
    def __init__(self, channel, reduction=6):
        super(SELayer, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(0, inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid())
        self.fco = nn.Linear(channel, channel, bias=True)
        self.leakyrelu = nn.LeakyReLU(0.02, inplace=True)

    def forward(self, input):
        y1 = self.fc1(input)
        y2 = self.leakyrelu(self.fco(torch.mul(y1, input))+input)
        return y2
    
class SELayer_w(nn.Module):
    def __init__(self, channel, reduction=6):
        super(SELayer_w, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(0, inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid())
        self.fco = nn.Linear(channel, channel, bias=True)

    def forward(self, input):
        y1 = self.fc1(input)
        y2 = self.fco(torch.mul(y1, input))+input
        return y2
    
# Branch networks
class Branch1(nn.Module):
    # branch net1
    def __init__(self, num_sensors):
        super(Branch1, self).__init__()
        self.numsensors = num_sensors
        self.net = self.__NET__(128, 6)
    
    def __NET__(self, nc, nb):
        layers = []
        layers.append(nn.Linear(self.numsensors, nc, bias=False))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        for i in range(nb):
            layers.append(SELayer(nc))
        layers.append(nn.Linear(nc, 64, bias=True))
        return nn.Sequential(*layers)
        
    def forward(self, input):
        output = self.net(input)
        return output
    
class Branch2(nn.Module):
    # branch net2
    def __init__(self, num_sensors):
        super(Branch2, self).__init__()
        self.numsensors = num_sensors
        self.net = self.__NET__(128, 6)
    
    def __NET__(self, nc, nb):
        layers = []
        layers.append(nn.Linear(7, nc, bias=False))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        for i in range(nb):
            layers.append(SELayer(nc))
        layers.append(nn.Linear(nc, 64, bias=True))
        return nn.Sequential(*layers)
        
    def forward(self, input):
        output = self.net(input)
        return output
    
class Branch3(nn.Module):
    # branch net3
    def __init__(self, num_sensors):
        super(Branch3, self).__init__()
        self.numsensors = num_sensors
        self.net = self.__NET__(128, 3)
    
    def __NET__(self, nc, nb):
        layers = []
        layers.append(nn.Linear(1, nc, bias=False))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        for i in range(nb):
            layers.append(SELayer(nc))
        layers.append(nn.Linear(nc, 64, bias=True))
        return nn.Sequential(*layers)
        
    def forward(self, input):
        output = self.net(input)
        return output
    
class Branch(nn.Module):
    def __init__(self, num_sensors):
        super(Branch, self).__init__()
        self.branch1 = Branch1(num_sensors)
        self.branch2 = Branch2(num_sensors)
        self.branch3 = Branch3(num_sensors)
        self.se = SELayer_w(64)
        self.leakyrelu = nn.LeakyReLU(0.02, inplace=True)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, input1, input2, input3):
        output1 = self.branch1(input1)
        output2 = self.branch2(input2)
        output3 = self.branch3(input3)
        output23 = self.leakyrelu(self.se(torch.mul(output2, output3)+0.001))
        
        output = torch.mean(output1*output23, dim=-1)+0.001
        return output
