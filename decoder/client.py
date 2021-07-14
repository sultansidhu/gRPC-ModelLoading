import grpc
import proto.python.proto.service.service_pb2_grpc as pb2_grpc
import proto.python.proto.service.service_pb2 as pb2
import onnx

from decoder import ProtoDecoder


class ModelEncodeClient(object):
    """
    Client for gRPC functionality
    """

    def __init__(self):
        self.host = '35.184.206.225'
        self.server_port = 8888

        # instantiate a channel
        self.channel = grpc.insecure_channel(
            '{}:{}'.format(self.host, self.server_port))

        # bind the client and the server
        self.stub = pb2_grpc.ModelEncodeStub(self.channel)

    def get_url(self, message):
        """
        Client function to call the rpc for GetServerResponse
        """
        message = pb2.Request(message=message)
        response = self.stub.GetEncodedModel(message)
        encoded_model = response.model
        decoder = ProtoDecoder(encoded_model)
        return decoder.decode_model_layers()



if __name__ == '__main__':
    client = ModelEncodeClient()
    decoded_model = client.get_url(message="") # empty message, since the server does not need it
    onnx.onnx.checker.check_model(decoded_model)
    print("ONNX model checked and decoded!")
