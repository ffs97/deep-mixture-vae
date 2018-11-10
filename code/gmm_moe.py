import numpy as np
import math
from scipy.stats import multivariate_normal as mn

from includes.utils import load_data, Dataset
from includes.config import Config

from tensorflow.examples.tutorials.mnist import input_data

from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

from tqdm import tqdm

class GMMMoE:
    def __init__(self, N, M, K, alpha=None, mean=None, sigma=None):
        '''
        N: Number of points in dataset
        M: Dimension of each X[i]
        K: Number of classes
        '''
        self.N = N
        self.M = M
        self.K = K

        self.alphas = alpha[np.newaxis, :]
        self.means = mean
        self.sigmas = sigma

        self.phis = np.random.normal(size=(K, M, K)) / np.sqrt(M)
        self.regularizer = np.identity(self.M) * 1e-4

    def gaussian_pdf(self, x, mu, sigma):
        '''
        x -> shape is (n, d)
        mu -> shape is either (n, d) or (1, d)
        sigma -> (d, d)
        '''
        diff = x - mu
        sigma_inv = np.linalg.inv(sigma)
        prob = np.sum(np.multiply(np.dot(diff, sigma_inv), diff), axis=1)

        denominator = np.sqrt(abs(np.linalg.det(sigma) + 1e-4))
        res = np.exp(-0.5 * prob) / denominator
        return res

    def E_step(self, X, Y):
        '''
        Shape of X: (num_samples X num_dimensions) = (N, M)
        Shape of Y: (num_samples X num_classes)    = (N, K)

        Returns H -> (num_samples X num_classes)   = (N, K)
        '''

        H = np.zeros((self.N, self.K))

        for e in range(self.K):
            # compute P(X | v_j)
            p_x_v = self.gaussian_pdf(X, self.means[e, None, :], self.sigmas[e, :, :]).reshape(self.N, 1)

            # compute P(Y | X, theta_j)
            Y_pred = np.dot(X, self.phis[e, :, :])
            p_y_x = self.gaussian_pdf(Y, Y_pred, np.identity(self.K)).reshape(self.N, 1)

            h = self.alphas[0, e] * p_x_v * p_y_x
            H[:, e] = h[0, :]

        # normalize along all experts
        sum_along_rows = np.sum(H, axis=1, keepdims=True)
        H = H / sum_along_rows
        return H

    def M_step(self, H, X, Y):
        H_sum = np.sum(H, axis=0, keepdims=True)
        self.alphas = H_sum / self.N
        print(self.alphas)

        for e in range(self.K):
            h_sum = H_sum[:, e] + 1e-4
            # mu estimate with shape (1, num_dimensions)
            mu_e = np.dot(H[:, None, e].T, X) / h_sum

            # sigma estimate
            diff = X - mu_e
            sigma_e = np.dot(np.multiply(H[:, None, e].T, diff.T), diff) / h_sum

            # weights of experts
            diag = np.multiply(H[:, None, e], np.identity(self.N))
            xT_diag = np.dot(X.T, diag)
            inverse_term = np.linalg.inv(np.dot(xT_diag, X) + self.regularizer)
            phi_e = np.dot(inverse_term, np.dot(xT_diag, Y))

            self.means[e, :] = mu_e
            self.sigmas[e, :] = sigma_e + self.regularizer
            self.phis[e, :] = phi_e

    def fit(self, X, Y, max_iter=100):
        # convert Y from class labels to one hot
        Y_binary = np.zeros((self.N, self.K))
        for i, j in enumerate(Y):
            Y_binary[i][j] = 1

        for idx in tqdm(range(max_iter)):
        # for idx in range(max_iter):
            H = self.E_step(X, Y_binary)
            self.M_step(H, X, Y_binary)


    def predict(self, X):
        Y_pred = 0
        for e in range(self.K):
            pred = np.dot(X, self.phis[e, :])
            weighted_y = self.alphas[0, e] * pred
            Y_pred += weighted_y

        return np.argmax(Y_pred, axis=1)



if __name__ == "__main__":
    N = 1000
    M = 784
    K = 10

    # pca = PCA(n_components=M)

    mnist = input_data.read_data_sets("data/mnist/", one_hot=False)
    X = mnist.train.images[:N]
    # X = pca.fit_transform(X)
    Y = mnist.train.labels[:N]

    # https://www.kaggle.com/danielhanchen/gaussian-mixture-models-on-mnist-iris#
    gmm = GaussianMixture(n_components=K, init_params='kmeans',
           n_init=5, max_iter=500)
    gmm.fit(X, Y)
    alpha, mu, sigma = gmm.weights_, gmm.means_, gmm.covariances_

    model = GMMMoE(N, M, K, alpha, mu, sigma)
    model.fit(X, Y)

    N_test = 1000
    X_test = mnist.test.images[: N_test]
    # X_test = pca.fit_transform(X_test)
    Y_test = mnist.test.labels[: N_test]
    Y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)
    print(accuracy)
