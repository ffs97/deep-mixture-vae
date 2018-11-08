import numpy as np
import math
from scipy.stats import multivariate_normal as mn

from includes.utils import load_data, Dataset
from includes.config import Config

from tensorflow.examples.tutorials.mnist import input_data
from sklearn.mixture import GaussianMixture

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

        # print(alpha.shape, mean.shape, sigma.shape)
        self.alphas = alpha[np.newaxis, :]
        self.means = mean
        self.sigmas = sigma

        self.phis = np.zeros((K, M, K))

    def gaussian_pdf(self, x, mu, sigma):
        # diff = x - mu
        pass


    def E_step(self, X, Y):
        H = np.zeros((self.N, self.K))

        for e in range(self.K):
            # compute P(X | v_j)
            p_x_v = self.gaussian_pdf(X, self.means[e, :, None], self.sigmas[e, :, :])

            # compute P(Y | X, theta_j)
            Y_pred = np.dot(X, self.phis[e, :, :])
            p_y_x = np.mean((Y - Y_pred)**2, axis=1, keepdims=True)

            assert p_x_v.shape == p_y_x.shape

            h = self.alphas[0, e] * p_x_v * p_y_x
            H[:, e] = h[0, :]

        # normalize along all experts
        sum_along_rows = np.sum(H, axis=1, keepdims=True) + 10e-4
        H = H / sum_along_rows
        return H

    def M_step(self, H, X, Y):
        H_sum = np.sum(H, axis=0, keepdims=True)
        self.alphas = H_sum / self.K

        for e in range(self.K):
            # mu estimate
            mu_e = np.sum(np.multiply(X, H[:, None, e]), axis=0, keepdims=True) / H_sum[:, e]

            # sigma estimate
            diff = X - mu_e
            sigma_e = np.dot(np.multiply(H[:, None, e].T, diff.T), diff)

            # weights of experts
            diag = np.multiply(H[:, None, e], np.identity(self.N))
            xT_diag = np.dot(X.T, diag)
            inverse_term = np.linalg.inv(np.dot(xT_diag, X))
            phi_e = np.dot(inverse_term, np.dot(xT_diag, Y))

            self.means[e, :] = mu_e
            self.sigmas[e, :] = sigma_e
            self.phis[e, :] = phi_e

    def fit(self, X, Y, max_iter=100):
        for idx in range(max_iter):
            print(idx)
            H = self.E_step(X, Y)
            self.M_step(H, X, Y)

if __name__ == "__main__":
    mnist = input_data.read_data_sets("data/mnist/", one_hot=True)
    X_train = mnist.train.images
    Y_train = mnist.train.labels

    N = 1000
    X = np.array(X_train[: N]).T
    Y = np.array(Y_train[: N]).T
    M = 784
    K = 10

    gmm = GaussianMixture(10)
    gmm.fit(X, Y)
    alpha, mu, sigma = gmm.weights_, gmm.means_, gmm.covariances_

    model = GMMMoE(N, M, K, alpha, mu, sigma)
    X = np.random.normal(size=(N, M))
    Y = np.random.randint(0, K, N)
    Y_binary = np.zeros((N, K))

    for i in range(N):
        Y_binary[i][Y[i]] = 1

    model.fit(X, Y_binary)
