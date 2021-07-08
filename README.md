# gRPC-ModelLoading
## Implementation Plan
### Overarching Desired Features 
- Simple API to use with ML Models
- Capability to use gRPC to send model data
- Sends over the basic information required to re-instantiate model on the other end. Includes:
    - Model parameters
    - Model hyperparams
    - Details for the model format, and which framework has been used on server side
- ML Framework agonostic
### Components 
**Note: For now, development will focus on simple PyTorch models only**
- Component to encode proto files for the state dictionary of model 
- Component to encode proto files for the peripheral model information, that stores model library, model layer names, etc.
- Component to take in a given PyTorch model, generate the previous two files from it 
- Component to send the data over a network using gRPC 
- Component to send the receive, parse the data over a network
- Component to restructure the model on the client side 
**This list is subject to change**
