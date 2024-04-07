import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_swiss_roll
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_openml


def load_MNIST(train_size = 1000, test_size = 100):
    assert train_size > test_size, "Train size must be greater than test size"

    X, y = fetch_openml('mnist_784', version = 'active', return_X_y = True, as_frame = False)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    y = y.astype(int)

    X = X[:train_size + test_size, :]
    y = y[:train_size + test_size]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = test_size, random_state = 123, stratify = y)

    return X_train, X_test, y_train, y_test

def reshape_to_desired_shape(vector, shape):
    desired_size = np.prod(shape)

    if len(vector) > desired_size:
        padded_vector = vector[:desired_size]  # Truncate if longer than desired size
    else:
        padded_vector = vector

    reshaped_matrix = padded_vector.reshape(shape)

    return reshaped_matrix



def plot_images(images, title, figsize=(10, 2)):
    fig, axs = plt.subplots(1, 5, figsize=figsize, sharex=True, sharey=True)
    axs = axs.flatten()

    for i, img in enumerate(images):
        axs[i].imshow(img, cmap='Greys')

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(title)
    plt.subplots_adjust(top=0.85)  # Adjust the top spacing
    plt.tight_layout()
    plt.show()

def report_results(X_test, y_test, RNA):
    y_hat = RNA.predict(X_test, y_test)

    RNA.plot_architecture()

    dim_orig = int(X_test[0].shape[0] ** 0.5)
    plot_images(X_test[:5].reshape(-1, dim_orig, dim_orig), f"Original image - ({dim_orig}x{dim_orig})",
                figsize = (10, 2))


    plot_images(y_hat[:5].reshape(-1, dim_orig, dim_orig), f"Reconstructed image - ({dim_orig}x{dim_orig})",
                figsize = (10, 2))

def report_encode(X_test, y_test, RNA):
    y_hat = RNA.predict(X_test, y_test)
    RNA.plot_architecture()

    dim_orig = int(X_test[0].shape[0]**0.5)
    plot_images(X_test[:5].reshape(-1, dim_orig, dim_orig), f"Original image - ({dim_orig}x{dim_orig})", figsize=(10, 2))

    # Plot encoded images
    encoded_dim = int(RNA.encoded_data.shape[1] ** 0.5)
    encoded_images = [reshape_to_desired_shape(RNA.encoded_data[i], (encoded_dim, encoded_dim)) for i in range(5)]
    plot_images(encoded_images, f"Encoded image - ({encoded_dim}x{encoded_dim})", figsize=(10, 2))


    plot_images(y_hat[:5].reshape(-1, dim_orig, dim_orig), f"Reconstructed image - ({dim_orig}x{dim_orig})", figsize=(10, 2))