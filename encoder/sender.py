"""
This file is an experimental file that will contain the functionality of defining PyTorch model, 
converting it to ONNX, encoding it, and then sending it over a gRPC network. 
"""
import torch.nn as nn
import torch.onnx
import onnx
    
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

if __name__ == "__main__":
    net = Net()


