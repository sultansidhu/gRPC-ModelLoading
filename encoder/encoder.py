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
from decoder.decoder import ProtoDecoder

class ProtoEncoder:
    """
    An encoder class that takes a model, encodes its layers and parameters 
    """
    def __init__(self, model: nn.Module, optim: optim = None, loss = None) -> None:
        self.model = model
        self.optim = optim
        self.loss = loss

    def encode_layer(self, layer:nn.Module, layers:ModelLayers) -> Layer:
        """
        A function to encode individual layers of a PyTorch model

        Args:
            layer (nn.Module): The layer that is to be encoded using Protobuf

        Returns:
            Layer: Encoded layer type for the input layer
        """
        # creates a new layer to be added to collection
        layer_proto = Layer()

        # based on layer type, set the Layer instance's type and features 
        if isinstance(layer, nn.Linear):
            layer_proto.type.append(Layer.LayerType.LINEAR)
            lin_layer = layers.LinearLayer()
            lin_layer.inFeatures.append(layer.in_features)
            lin_layer.outFeatures.append(layer.out_features)
            layers.linearLayers.append(lin_layer)
        elif isinstance(layer, nn.ReLU):
            layer_proto.type.append(Layer.LayerType.RELU)
            relu_layer = layers.ReLULayer()
            relu_layer.inPlace.append(layer.inplace)
            layers.reluLayers.append(relu_layer)
        elif isinstance(layer, nn.LogSoftmax):
            layer_proto.type.append(Layer.LayerType.LOGSOFTMAX)
            logsoftmax_layer = layers.LogSoftmaxLayer()
            logsoftmax_layer.dim.append(layer.dim)
            layers.logsoftmaxLayers.append(logsoftmax_layer)
        else:
            print(f"Warning: Layer {layer} does not have a ProtoBuf Mapping. Program will exit.")
            exit(1)
        return layer_proto

    def encode_model_layers(self) -> str:
        """
        A function that encodes a given PyTorch model's layers using Protobuf.

        Returns:
            str: An encoded byte-string which contains within it information regarding
            the layers of the encoded model. 
        """
        layers = ModelLayers()
        for layer in self.model.children():
            encoded_layer = self.encode_layer(layer, layers)
            layers.layers.append(encoded_layer)
        return layers.SerializeToString()





    


