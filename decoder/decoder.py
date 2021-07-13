"""
Decoder file for ML Models for the Fledge Federated Learning Framework. 
Currently, development work taking place only for encoding PyTorch models with protobuf, and sending
them over a network using gRPC. 
"""
# module imports
from collections import OrderedDict

# torch imports
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Imports for layer types.
from proto.python.proto.layers.layers_pb2 import ModelLayers, Layer

class ProtoDecoder:
    """
    A decoder class, that takes in the generated byte-string from the Encoder's encode_model_layers function
    and returns a reconstructed model on the client side. 
    """
    def __init__(self, encoded_model: str) -> None:
        self.encoded_model = encoded_model
    
    def decode_model_layers(self) -> OrderedDict:
        """
        A function that decodes the encoded model byte-string, generated using protobuf.

        Returns:
            OrderedDict: Returns an ordered dictionary of the module layers involved within the model.
        """
        # initialize the layer list, parse the string
        model_layers = []
        encoded_layers = ModelLayers()
        encoded_layers.ParseFromString(self.encoded_model)
        layer_counter = 0

        # for each type of layer encountered, parse its properties
        for layer in encoded_layers.layers:
            layer_counter += 1
            layer_type = layer.type.pop(0)
            if layer_type == Layer.LayerType.LINEAR:
                lin_layer = encoded_layers.linearLayers.pop(0)
                in_features = lin_layer.inFeatures.pop(0)
                out_features = lin_layer.outFeatures.pop(0)
                decoded_layer = nn.Linear(in_features, out_features)
            elif layer_type == Layer.LayerType.RELU:
                relu_layer = encoded_layers.reluLayers.pop(0)
                inplace = relu_layer.inPlace.pop(0)
                decoded_layer = nn.ReLU(inplace)
            elif layer_type == Layer.LayerType.LOGSOFTMAX:
                logsoftmax_layer = encoded_layers.logsoftmaxLayers.pop(0)
                decoded_layer = nn.LogSoftmax(logsoftmax_layer.dim.pop(0))
            else:
                print(f"Error: Encountered layer without available conversion {layer_type}. Please consult layers.proto file. Exiting.")
                exit(1)
            model_layers.append(decoded_layer)

        # check for model depth being consistent with layers decoded
        assert layer_counter == len(model_layers)
        return model_layers


if __name__ == "__main__":
    class Hello:
        def __init__(self) -> None:
            pass
        def forward(self, x):
            return x + 2
    
    def forward4(x):
        return x + 3
    h = Hello()
    print(h.forward(2))
    h.forward = forward4
    print(h.forward(2))

