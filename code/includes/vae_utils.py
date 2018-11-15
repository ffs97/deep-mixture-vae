import math
import numpy as np
import pickle as pkl
import scipy.sparse as sp


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_gumbel(shape, eps=1e-20):
    U = np.random.uniform(0, 1, shape)
    return - np.log(eps - np.log(U + eps))


def load_data(datagroup, **args):
    def spiral(N_tr=5000, N_ts=1000):
        D = 2
        K = 5
        X_tr = np.zeros((N_tr * K, D))
        X_ts = np.zeros((N_ts * K, D))

        for j in range(K):
            ix = range(N_tr * j, N_tr * (j + 1))
            r = np.linspace(2.5, 10.0, N_tr)
            t = np.linspace(j * 1.25, (j + 1) * 1.25, N_tr) + \
                np.random.randn(N_tr) * 0.05
            X_tr[ix] = np.c_[r * np.sin(t), r * np.cos(t)]

        for j in range(K):
            ix = range(N_ts * j, N_ts * (j + 1))
            r = np.linspace(2.5, 10.0, N_ts)
            t = np.linspace(j * 1.25, (j + 1) * 1.25, N_ts) + \
                np.random.randn(N_ts) * 0.05
            X_ts[ix] = np.c_[r * np.sin(t), r * np.cos(t)]

        return X_tr, X_ts

    def mnist(dir="data/mnist"):
        from tensorflow.examples.tutorials.mnist import input_data

        mnist = input_data.read_data_sets(dir, one_hot=True)

        test_data = mnist.test.images
        train_data = mnist.train.images

        return train_data, test_data

    if datagroup == "spiral":
        return spiral(**args)
    elif datagroup == "mnist":
        return mnist(**args)
    else:
        assert(False)


class Dataset:
    def __init__(self, data, batch_size=100, shuffle=True):
        self.data = np.copy(data)
        self.batch_size = batch_size

        self.data_dim = self.data.shape[1]

        self.epoch_len = int(math.ceil(len(data) / batch_size))

        if shuffle:
            np.random.shuffle(self.data)

    def get_batches(self, shuffle=True):
        if shuffle:
            np.random.shuffle(self.data)

        batch = []
        for row in self.data:
            batch.append(row)
            if len(batch) == self.batch_size:
                yield np.array(batch)
                batch = []
        if len(batch) > 0:
            yield np.array(batch)

    def __len__(self):
        return self.epoch_len
