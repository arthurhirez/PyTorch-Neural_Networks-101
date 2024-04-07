import time
import pickle

import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:

    def __init__(self, *layers, lr = 1e-2):
        self.layers = []
        self.lr = lr

        for layer in layers: self.add(layer)

    def __str__(self):
        return f"NeuralNetwork:\nLayer: fazer algo\nLearnable parameters count: {self.params_count}\n\n"

    def predict(self, X_test, y_test = None, *args, **kwargs):
        y_hat = self.forward(X_test)
        self.size_test = y_test.shape[0]
        if self.layers[-1].activation == 'linear':
            min_val = np.min(y_hat, axis = 0)
            max_val = np.max(y_hat, axis = 0)
            normalized_data = (y_hat - min_val) / (max_val - min_val)
            if (y_test is not None):
                self.mse_test = self.mean_squared_error(y_test, y_hat)
            return normalized_data

        if(y_test is not None):
            self.mse_test = self.mean_squared_error(y_test, y_hat)
        return y_hat

    def mean_squared_error(self, y_true, y_pred, prime = False):
        '''
        confirmar shape!!!!!!!!!!
        '''
        if prime:
            # Matrix -> gradient
            return (2 * (y_pred - y_true)) / y_true.shape[1]
            # Scalar -> MSE batch/example
        return np.mean(np.power(y_true - y_pred, 2))
        # return np.mean(np.power(y_true - y_pred, 2), axis=1)

    def params_count(self):
        return [module.params_info for module in self.layers]

    def add(self, module):
        '''fazer get e set para input output no module'''
        if len(self.layers):
            assert self.layers[-1].out_features == module.in_features, "Shape inconsistency adding new layer!"

        if self.lr: module.set_lr(self.lr)
        self.layers.append(module)

    def forward(self, input_data):
        prediction = input_data
        for module in self.layers:
            prediction = module.forward(prediction)
        return prediction

    def backward(self, received_grad):
        error_grad = received_grad
        for module in reversed(self.layers):
            error_grad = module.backward(error_grad)
        return error_grad

    def fit(self, samples, targets,
            epochs = 100, batch_size = 100,
            shuffle = True,
            verbose = False):

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
                y_hat = self.forward(X_batch)
                loss += self.mean_squared_error(y_batch, y_hat, prime = False)

                loss_grad = self.mean_squared_error(y_batch, y_hat, prime = True)
                _ = self.backward(loss_grad)

            mse = loss / i
            self.err_trace.append(mse)
            if epoch % 50 == 0 and verbose: print(f"{epoch + 1}/{epochs}, error={mse:.8f}")


        self.time_train = time.time() - self.time_train
        self.mse_train = mse
        print('{:^10}|{:^20}|{:^20}|').format(f'{epoch + 1}/{epochs}',
                                             f'time_elapsed: {self.time_train:.2f}',
                                              f'error={mse:.8f}')

    def minibatch_generator(self, X, y, batch_size, shuffle = True):
        idxs = np.arange(X.shape[0])
        if shuffle:
            np.random.shuffle(idxs)

        num_batches = (X.shape[0] + batch_size - 1) // batch_size  # Calculate number of batches

        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, X.shape[0])  # Determine end index for the batch

            batch_idx = idxs[start_idx:end_idx]
            yield X[batch_idx], y[batch_idx]

    def plot_architecture(self):
        num_layers = len(self.layers)

        header = ['Layer', 'Previous Neurons', 'Current Neurons', 'Activation Function', 'Parameters Info']
        separator = '+' + '+'.join(['-' * 20] * (len(header) - 1) + ['-' * 30]) + '+'

        if (self.mse_test is None):
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

        for i in range(num_layers):
            layer = self.layers[i]
            current_neurons = layer.out_features
            previous_neurons = layer.in_features
            activation_function = layer.activation if hasattr(layer, 'activation') else 'None'
            params_w, params_b = layer.params_info()
            params_info_str = f"({params_w}) + ({params_b}) = {params_w + params_b}"
            print(('{:^21}|{:^20}|{:^20}|{:^20}|{:^30}').format(i + 1, previous_neurons, current_neurons,
                                                              activation_function, params_info_str))

    def plot_loss(self):

        plt.figure(figsize = (10, 6))
        plt.plot(self.err_trace, marker = 'o', linestyle = '-')
        plt.title('Epoch Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)