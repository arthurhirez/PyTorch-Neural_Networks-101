import time
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from NeuralNetwork import NeuralNetwork
from Dense import Dense


class Autoencoder(NeuralNetwork):

    def __init__(self, layers = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = []
        self.decoder = []

        if layers is not None:
            for enc in layers[0]:
                if len(self.layers): assert self.encoder[
                                                -1].out_features == enc.in_features, "Shape inconsistency adding new layer!"
                if self.lr: enc.set_lr(self.lr)
                self.encoder.append(enc)

            for dec in layers[1]:
                if len(self.layers): assert self.decoder[
                                                -1].out_features == dec.in_features, "Shape inconsistency adding new layer!"
                if self.lr: dec.set_lr(self.lr)
                self.decoder.append(dec)

        self.encoded_data = None

    def add(self, module):
        track_dim = []
        if len(self.decoder): print("Deleting previous architecture.")
        if len(self.encoder): assert self.encoder[
                                         -1].out_features == module.in_features, "Shape inconsistency adding new layer!"
        if self.lr: module.set_lr(self.lr)
        self.encoder.append(module)


        if (module.output_layer):
            module.set_act("linear")
            for module in reversed(self.encoder):
                output_layer, activation, in_features, out_features = module.get_init()
                self.decoder.append(Dense(out_features, in_features, output_layer, activation))
            self.decoder[0].output_layer = False
            self.decoder[0].set_act("relu")
            self.decoder[-1].output_layer = True
            self.decoder[-1].set_act("sigmoid")

    def predict(self, X_test, y_test = None, isimg = True, *args, **kwargs):
        y_hat = self.forward(X_test)

        # if isimg and self.layers[-1].activation == 'linear':
        #     scaler = MinMaxScaler()
        #     y_hat = scaler.fit_transform(y_hat)

        if (y_test is not None):
            self.size_test = y_test.shape[0]
            self.mse_test = self.mean_squared_error(y_test, y_hat)
        return y_hat

    def forward(self, input_data, isimg = True):


        prediction = input_data
        for module in self.encoder:
            prediction = module.forward(prediction)

        self.encoded_data = prediction

        for module in self.decoder:
            prediction = module.forward(prediction)

        # if self.decoder[-1].activation == 'linear':
        #     scaler = MinMaxScaler() if isimg else StandardScaler()
        #     prediction = scaler.fit_transform(prediction)
        # return normalized_data

        return prediction

    def backward(self, received_grad):
        # received_grad = self.scaler.inverse_transform(received_grad)
        error_grad = received_grad
        for module in reversed(self.decoder):
            error_grad = module.backward(error_grad)

        for module in reversed(self.encoder):
            error_grad = module.backward(error_grad)

        return error_grad

    def fit(self, samples, targets,
            epochs = 100, batch_size = 100,
            shuffle = True,
            verbose = False,
            isimg = True):

        # Criar um dicionário de estado talvez fosse melhor
        self.batch = batch_size
        self.epochs = epochs
        self.size_train = samples.shape[0]
        self.err_trace = []
        self.time_train = time.time()

        for epoch in range(epochs):
            loss = 0
            batch_gen = self.minibatch_generator(samples, targets, batch_size, shuffle)

            for i, (X_batch, y_batch) in enumerate(batch_gen):
                y_hat = self.forward(X_batch, isimg)
                loss += self.mean_squared_error(y_batch, y_hat, prime = False)

                loss_grad = self.mean_squared_error(y_batch, y_hat, prime = True)
                _ = self.backward(loss_grad)

            # print(loss.shape)
            mse = loss / i
            self.err_trace.append(mse)

            if epoch % 50 == 0 and verbose: print(f"{epoch + 1}/{epochs}, error={mse:.8f}")

        self.time_train = time.time() - self.time_train
        self.mse_train = mse
        print(('{:^10}|{:^20}|{:^20}|').format(f'{epoch + 1}/{epochs}',
                                              f'time_elapsed: {self.time_train:.2f}',
                                              f'error={mse:.8f}'))


    def plot_architecture(self):

        header = ['Layer', 'Previous Neurons', 'Current Neurons', 'Activation Function', 'Parameters Info']
        separator = '+' + '+'.join(['-' * 20] * (len(header) - 1) + ['-' * 30]) + '+'

        if(self.mse_test is None):
            print("Attention! Run the predict() with the test data to compute the MSE_test")
            self.size_test = -1
            self.mse_test = -1
        print(separator)
        print('{:^21}|{:^20}|{:^20}|{:^20}|{:^30}'.format(f"Train Size: {self.size_train}",
                                                          f"Test Size: {self.size_test}",
                                                          f"Train MSE: {self.mse_train:.5f}",
                                                          f"Test MSE: {self.mse_test:.5f}",
                                                          f"Epochs: {self.epochs} | lr: {self.lr}"))
        print(separator)
        print(separator)

        print('{:^21}|{:^20}|{:^20}|{:^20}|{:^30}'.format(*header))  # Adjusted header alignment
        print(separator)

        for i in range(len(self.encoder)):
            layer = self.encoder[i]
            current_neurons = layer.out_features
            previous_neurons = layer.in_features
            activation_function = layer.activation if hasattr(layer, 'activation') else 'None'
            params_w, params_b = layer.params_info()
            params_info_str = f"({params_w}) + ({params_b}) = {params_w + params_b}"
            print('{:^21}|{:^20}|{:^20}|{:^20}|{:^30}'.format(f'Encoder:{i + 1}', previous_neurons, current_neurons,
                                                              activation_function, params_info_str))

        for i in range(len(self.decoder)):
            layer = self.decoder[i]
            current_neurons = layer.out_features
            previous_neurons = layer.in_features
            activation_function = layer.activation if hasattr(layer, 'activation') else 'None'
            params_w, params_b = layer.params_info()
            params_info_str = f"({params_w}) + ({params_b}) = {params_w + params_b}"
            print('{:^21}|{:^20}|{:^20}|{:^20}|{:^30}'.format(f'Decoder:{i + 1}', previous_neurons, current_neurons,
                                                              activation_function, params_info_str))

