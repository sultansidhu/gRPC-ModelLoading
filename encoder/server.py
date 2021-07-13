# module imports
import grpc
from concurrent import futures
import time

# import Torch libraries for example network
import torch.nn as nn

# import protobuf created files
import proto.python.proto.service.service_pb2_grpc as pb2_grpc
import proto.python.proto.service.service_pb2 as pb2

# importing encoder to encode model, and send
from encoder.encoder import ProtoEncoder


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

class ModelEncodeService(pb2_grpc.ModelEncodeServicer):

    def __init__(self, model, *args, **kwargs):
        self.model = model
        self.encoder = ProtoEncoder(self.model)

    def GetServerResponse(self, request, context):
        start_time = time.time()
        encoded_model = self.encoder.encode_model_layers()
        result = {'model': encoded_model, 'received': True}
        end_time = time.time()
        print(f"Time taken for returning response - {end_time - start_time}")
        return pb2.MessageResponse(**result)


def serve():
    net = Net()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_ModelEncodeServicer_to_server(ModelEncodeService(net), server)
    server.add_insecure_port('[::]:8888')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()