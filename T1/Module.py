import numpy as np

class Module():
    '''
    Attributes:
        input_data  (X)       := features or results of activation function   (m, i)
        output_data (Z)       := results of net sum or activation function    (j, m)

        received_grad (dE/dZ) := Gradient received to adjust weights          (j, m)
        retro_grad    (dE/dX) := Gradient to be passed to previous layer      (m, i)

        n_samples              := for batch purposes                           int
        lr                     := learning rate                                float

    Indexes Conventions:
        i   := number of neurons previous layer
        j   := number of neurons current layer
        m   := batch size
    '''

    def __init__(self, lr = 1e-2, *args, **kwargs):
        # Data
        self.input_data = None
        self.output_data = None

        # Gradients
        self.retro_grad = None
        self.received_grad = None

        # Network
        self.n_samples = None
        self.lr = lr
        self.activation = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def params_info(self):
        # Return the # of trainable parameters
        raise NotImplementedError

    def set_lr(self, lr_new):
        self.lr = lr_new

    def set_act(self, actv_func):
        self.activation = actv_func

    def forward(self, input_data):
        self.input_data = input_data
        self._forward_propagation(input_data)
        return self.output_data

    def _forward_propagation(self, input_data):
        raise NotImplementedError

    def backward(self, upstream_grad):
        self.received_grad = upstream_grad
        self._backward_propagation(upstream_grad)
        return self.retro_grad

    def _backward_propagation(self, upstream_grad):
        raise NotImplementedError

    # ---------------------------------------------------------------- #
    # The following functions could be implemented as Classes (Optimizer and Activation_Functions) but for simplicity they're functions

    def optimizer_sgd(self, parameters, gradients):
        for param, grad in zip(parameters, gradients):
            param -= grad * self.lr

    '''
    ATENÇÃO NO SHAPE ATIVAÇÃO!
    '''


    def sigmoid(self, inputs, prime = False):
        f = 1 / (1 + np.exp(-inputs))
        if prime:
            return (f * (1 - f))
        return f

    def relu(self, inputs, prime = False):
        if prime:
            return (1. * (inputs > 0))
        return np.maximum(inputs, 0, out = inputs)

    def linear(self, inputs, prime = False):
        if prime: return np.ones(inputs.shape)
        return inputs

    def activate(self, inputs, prime = False):
        if self.activation == 'relu':
            return self.relu(inputs, prime)
        elif self.activation == 'sigmoid':
            return self.sigmoid(inputs, prime)
        return self.linear(inputs, prime)