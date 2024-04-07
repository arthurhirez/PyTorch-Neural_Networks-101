import numpy as np
from Module import Module


class Dense(Module):
    '''
    Attributes:
        in_features  (i)       := Number of neurons previous layer  int
        out_features (j)       := Number of neurons current layer    int

        self.net_input (Z)      := Result -> Z = W @ X + b       (j, m)
        self.weight    (W)      := Weights connecting to the previous layer  (j, i)
        self.bias      (b)      := Bias applied to the current layer (j, 1) -> broadcast

        output_layer    := Define if it's the last layer         bool


    Indexes Conventions:
        i   := number of neurons previous layer
        j   := number of neurons current layer
        m   := batch size
    '''

    def __init__(self, in_feat, out_feat, output_layer = False, actv_func = 'relu', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_layer = output_layer
        self.activation = 'linear' if output_layer else actv_func

        self.in_features = in_feat
        self.out_features = out_feat

        self._initialize_parameters()

    def get_init(self):
        return self.output_layer, self.activation, self.in_features, self.out_features

    def _initialize_parameters(self):
        a = np.sqrt(2 / self.in_features)
        self.weight = np.random.normal(loc = 0, scale = a,
                                       size = (self.out_features, self.in_features))

        self.bias = np.zeros(self.out_features)

    def params_info(self):
        return int(self.weight.size), int(self.bias.size)

    def _forward_propagation(self, input_data):
        # assert self.input_data.shape[0] == self.in_features, f"Input size doesn't match: {self.input_data.shape[0]} ≠ {self.in_features}"

        self.net_input = np.dot(input_data, self.weight.T) + self.bias
        self.output_data = self.activate(inputs = self.net_input, prime = False)

    def _backward_propagation(self, received_grad):
        # assert self.received_grad.shape[0] == self.out_features, "Upstream gradients size doesn't match"

        '''
        ATENÇÃO NO SHAPE ATIVAÇÃO!
        '''

        activ_g = self.activate(inputs = self.net_input, prime = True)
        activation_grad = np.multiply(received_grad, activ_g)

        self.retro_grad = np.dot(activation_grad, self.weight)

        weights_error = np.dot(activation_grad.T, self.input_data)
        self.optimizer_sgd([self.weight, self.bias], [weights_error, np.sum(activation_grad, axis = 0)])



    def module_shapes(self):
        print("Shapes:")
        print("Input Data (X):        ", tuple(self.input_data.shape))
        print("Output Data (Z):       ", tuple(self.output_data.shape))
        print("Received Gradient (dE/dZ): ", tuple(self.received_grad.shape))
        print("Retro Gradient (dE/dX):    ", tuple(self.retro_grad.shape))
        print("Net Input (Z):         ", tuple(self.net_input.shape))
        print("Weight (W):            ", tuple(self.weight.shape))
        print("Bias (b):              ", tuple(self.bias.shape))