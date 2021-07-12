from enum import Enum
import torch.nn as nn
from proto.python.proto.layers.linear_pb2 import LinearLayer
from proto.python.proto.layers.logsoftmax_pb2 import LogSoftmax
from proto.python.proto.layers.relu_pb2 import ReLU

class Layers(Enum):
