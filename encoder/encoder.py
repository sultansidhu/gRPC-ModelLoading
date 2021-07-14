"""
Encoder file for ML Models for the Fledge Federated Learning Framework. 
Currently, development work taking place only for encoding PyTorch models with protobuf, and sending
them over a network using gRPC. 
"""
# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Imports for layer types.
from proto.python.proto.layers.layers_pb2 import ModelLayers, Layer
from onnx_manager.onnx_manager import ONNXManager
from decoder.decoder import ProtoDecoder

class ProtoEncoder:
    """
    An encoder class that takes a model, encodes its layers and parameters 
    """
    def __init__(self, model: nn.Module, optim: optim = None, loss = None) -> None:
        self.model = model
        self.optim = optim
        self.loss = loss
        self.manager = ONNXManager()

    def encode_model_layers(self) -> bytes:
        """
        A function that encodes a given PyTorch model's layers using Protobuf.

        Returns:
            bytes: An encoded byte-string which contains within it information regarding
            the layers of the encoded model. Encoded as ONNX.
        """
        return self.manager.encode(self.model, torch.randn((28*28, 28*28)))





    


