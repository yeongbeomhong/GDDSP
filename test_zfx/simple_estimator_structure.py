import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    MLP (Multi-layer Perception). 
    One layer consists of what as below:
        - 1 Dense Layer(=FC Layer)
        - 1 Layer Norm
        - 1 ReLU (Activation)
    constructor arguments :
        n_input : dimension of input : dimension of encoded audio feature to be fed into MLP of Decoder.
        n_units : dimension of hidden unit
        n_layer : depth of MLP (the number of layers)
        relu : relu (default : nn.ReLU, can be changed to nn.LeakyReLU, nn.PReLU for example.)
    input(x): torch.tensor w/ shape(B, ... , n_input)
    output(x): torch.tensor w/ (B, ..., n_units)
    """

    def __init__(self, n_input, n_units, n_layer, activation = "sigmoid", inplace=True):
        super().__init__()
        self.n_layer = n_layer # Depth of MLP
        self.n_input = n_input # Input size of MLP = (Batch, Frames, n_input)
        self.n_units = n_units # output size of MLP = (Batch, Frames, n_units)
        self.inplace = inplace
        
        if activation == "sigmoid": 
            self.add_module(
                f"mlp_layer1",
                nn.Sequential(
                    nn.Linear(n_input, n_units),
                    nn.LayerNorm(normalized_shape=n_units),
                    nn.Sigmoid()
                    #relu(inplace=self.inplace),
                ),
            )
            for i in range(2, n_layer + 1):
                self.add_module(
                    f"mlp_layer{i}",
                    nn.Sequential(
                        nn.Linear(n_units, n_units),
                        nn.LayerNorm(normalized_shape=n_units)
                    ),
                )
        if activation == "relu":
            self.add_module(
                f"mlp_layer1",
                nn.Sequential(
                    nn.Linear(n_input, n_units),
                    nn.LayerNorm(normalized_shape=n_units),
                    nn.ReLU()
                ),
            )
            for i in range(2, n_layer + 1):
                self.add_module(
                    f"mlp_layer{i}",
                    nn.Sequential(
                        nn.Linear(n_units, n_units),
                        nn.LayerNorm(normalized_shape=n_units)
                    ),
                )
    def forward(self, x):
        for i in range(1, self.n_layer + 1):
            x = self.__getattr__(f"mlp_layer{i}")(x)
        return x