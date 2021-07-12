"""
Encoder file for ML Models for the Fledge Federated Learning Framework. 
Currently, development work taking place only for encoding PyTorch models with protobuf, and sending
them over a network using gRPC. 
"""
# torch imports
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Imports for layer types.
from proto.python.proto.layers.layers_pb2 import ModelLayers, Layer

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
    def __init__(self, model: nn.Module, optim: optim, loss) -> None:
        self.model = model
        self.optim = optim
        self.loss = loss

    def encode_model_layers(self) -> str:
        """
        A function that encodes a given PyTorch model's layers using Protobuf.

        Returns:
            str: An encoded byte-string which contains within it information regarding
            the layers of the encoded model. 
        """
        layers = ModelLayers()
        for layer in self.model.children():
            # creates a new layer to be added to collection
            layer_proto = layers.layers.add() 

            # based on layer type, set the Layer instance's type and features 
            if isinstance(layer, nn.Linear):
                layer_proto.type = Layer.LayerType.LINEAR
                layer_proto.LinearLayer.inFeatures = layer.in_features
                layer_proto.LinearLayer.outFeatures = layer.out_features
                layer_proto.LinearLayer.bias = bool(layer.bias)
            elif isinstance(layer, nn.ReLU):
                layer_proto.type = Layer.LayerType.RELU
                layer_proto.ReLULayer.inPlace = layer.inplace
            elif isinstance(layer, nn.LogSoftmax):
                layer_proto.type = Layer.LayerType.LOGSOFTMAX
                layer_proto.LogSoftmaxLayer.dim = layer.dim
            else:
                print(f"Warning: Layer {layer} does not have a ProtoBuf Mapping. Program will exit.")
                exit(1)
        return layers.SerializeToString()


if __name__ == "__main__":
    # model = Net()
    # # we need to send model.state_dict() and module information from model.children()

    # # model.state_dict() can be encoded directly as a dictionary
    # for key in model.state_dict().keys():
    #     print(key)
    #     print(model.state_dict()[key])
    #     break

    # # model.children() can be encoded with their own protos, each proto corresponding to a layer
    # for layer in model.children():
    #     print(layer)
    #     print(f"{isinstance(layer, nn.Linear)}\n")
    #     print(layer.in_features)
    #     print(layer.out_features)
    #     break
    layers = ModelLayers()
    #print(layers)
    layer_proto = layers.layers.add() 
    #print(layer_proto)
    layer_proto.type = Layer.LayerType.RELU
    layer_proto.LinearLayer.inFeatures = 10
    layer_proto.LinearLayer.outFeatures = 15
    layer_proto.LinearLayer.bias = True
    #print(layer_proto)
    string = layers.SerializeToString()
    record = ModelLayers()
    record.ParseFromString(string)
    for layer in record.layers:
        print(layer.type)
        print(layer.LinearLayer.inFeatures)

    


