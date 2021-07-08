"""
Encoder file for ML Models for the Fledge Federated Learning Framework. 
Currently, development work taking place only for encoding PyTorch models with protobuf, and sending
them over a network using gRPC. 
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict

class Net(nn.Module):
    """
    A sample neural network, utilizing simple layers.

    Inherits from - nn.Module
    """
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(28*28, 512)
        self.sigmoid = nn.ReLU()
        self.output = nn.Linear(512, 10)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

class ProtoEncoder:
    """
    An encoder class that takes a model, encodes its layers and parameters 
    """

if __name__ == "__main__":
    model = Net()
    # we need to send model.state_dict() and module information from model.children()

    # model.state_dict() can be encoded directly as a dictionary
    for key in model.state_dict().keys():
        print(model.state_dict()[key])
        break

    # model.children() can be encoded with their own protos, each proto corresponding to a layer
    for layer in model.children():
        print(f"{isinstance(layer, nn.modules.linear.Linear)}\n")
        break
    


