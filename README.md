# gRPC-ModelLoading
## Implementation Plan
### Overarching Desired Features 
- Simple API to use with ML Models
- Capability to use gRPC to send model data
- Sends over the basic information required to re-instantiate model on the other end. Includes:
    - Model parameters
    - Model hyperparams
    - Details for the model format, and which framework has been used on server side
- ML Framework agnostic 

### Notes on this project
- ML Framework agnosticism comes from the usage of ONNX intermediary format.
- All ML models are currently envisioned to be converted to ONNX before network transmission.
- ONNX is convenient because the saving mechanism of ONNX models involves turning both, the parameters and layers into bytestrings.
- These bytestrings can then be sent over the network using gRPC in a very convenient fashion.

### Components 
**Note: For now, development will focus on simple PyTorch models only**
- Component to encode proto files for the state dictionary of model (DONE)
- Component to encode proto files for the peripheral model information, that stores model library, model layer names, etc. (DONE)
- Component to take in a given PyTorch model, generate the previous two files from it (DONE)
- Component to send the data over a network using gRPC (DONE)
- Component to receive, parse the data over a network (DONE)
- Component to restructure the model on the client side (DONE)
- Component to send user-defined instructions to the client side
- Component that can control training on the client side, and inference on the server side
- **We need to be able to figure out how we can train models using ONNX**
**This list is subject to change**
