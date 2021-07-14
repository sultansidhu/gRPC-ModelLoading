"""
This file is an experimental file that will contain the functionality of defining PyTorch model, 
converting it to ONNX, encoding it, and then sending it over a gRPC network and decoding it back
to an ONNX model. 
"""
import os
import tempfile
import torch
import torch.nn as nn
import torch.onnx
import onnx
from onnx2pytorch import ConvertModel


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


class ONNXManager:
    """
    A class responsible for managing the model as an ONNX import. 
    ONNX export writes a file as a binary protobuf, thus it can be read 
    and be sent over gRPC quite easily. 
    """
    def __init__(self) -> None:
        pass

    def encode(self, model: nn.Module, dummy_input: torch.Tensor):
        """
        The chief function for encoding a PyTorch model. 

        Args:
            model (nn.Module): The model to be exported as ONNX
            dummy_input (torch.Tensor): Dummy input required for ONNX
        """
        _, path = tempfile.mkstemp()

        try:
            torch.onnx.export(model, dummy_input, path, verbose=True)
            with open(path, "rb") as fd:
                converted_model = fd.read()
        except Exception as e:
            converted_model = None
            print(f"Error occurred: {e}\n")
        finally:
            os.remove(path)
            return converted_model
    
    def decode(self, model: bytes):
        """
        Function for decoding the received bytestring of the encoded ONNX model. 
        Decodes it back into a PyTorch model. 

        Args:
            model (bytes): bytestring containing the model description and parameters. 

        Returns:
            nn.Module: The decoded PyTorch model. 
        """
        _, path = tempfile.mkstemp()
        with open(path, "wb") as fd:
            fd.write(model)
        onnx_model = onnx.load(path)
        pytorch_model = ConvertModel(onnx_model)
        os.remove(path)
        return pytorch_model
        

if __name__ == "__main__":
    net = Net()
    dummy_input = torch.randn((28*28, 28*28))
    manager = ONNXManager()

    print(net)
    model_bytes = manager.encode(net, dummy_input)
    torch_model = manager.decode(model_bytes)
    print(torch_model)
