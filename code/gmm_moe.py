import numpy as np
import math
from scipy.stats import multivariate_normal as mn

class GMMMoE:
    def __init__(self, N, M, K):
        '''
        N: Number of points in dataset
        M: Dimension of each X[i]
        K: Number of classes
        '''
        self.N = N
        self.M = M
        self.K = K

        # TODO: Initialize properly
        alpha = np.random.uniform(0, 1, size=(1, K))
        self.alphas = alpha / alpha.sum(keepdims=True)
        self.means = np.random.normal(size=(M, K))
        self.sigmas = np.array([np.identity(M) for _ in range(K)])

        self.phis = np.random.normal(size=(K, M, K))

    def E_step(self, X, Y):
        H = np.zeros((self.N, self.K))

        for e in range(self.K):
            # compute P(X | v_j)
            p_x_v = mn.pdf(X, self.means[:, e], self.sigmas[e, :, :]).reshape(self.N, 1)

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
            mu_e = np.sum(np.multiply(X, H[:, None, e]), axis=0, keepdims=True) / H_sum[:, e]

            diff = X - mu_e
            s = np.dot(np.multiply(H[:, None, e].T, diff.T), diff)


            diag = np.multiply(H[:, None, e], np.identity(self.N))
            xT_diag = np.dot(X.T, diag)
            inverse_term = np.linalg.inv(np.dot(xT_diag, X))
            phi_e = np.dot(inverse_term, np.dot(xT_diag, Y))

            self.means[:, e] = mu_e
            self.phis[e, :] = phi_e

    def fit(self, X, Y, max_iter=100):
        for idx in range(max_iter):
            H = self.E_step(X, Y)
            self.M_step(H, X, Y)

if __name__ == "__main__":
    N = 100
    M = 15
    K = 10

    model = GMMMoE(N, M, K)
    X = np.random.normal(size=(N, M))
    Y = np.random.randint(0, K, N)
    Y_binary = np.zeros((N, K))

    for i in range(N):
        Y_binary[i][Y[i]] = 1

    model.fit(X, Y_binary)

