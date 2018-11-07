import math
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(datagroup, **args):
    def spiral(dataset="spiral", N_tr=5000, N_ts=1000):
        D = 2
        K = 5
        X_tr = np.zeros((N_tr * K, D))
        X_ts = np.zeros((N_ts * K, D))

        for j in xrange(K):
            ix = range(N_tr * j, N_tr * (j + 1))
            r = np.linspace(2.5, 10.0, N_tr)
            t = np.linspace(j * 1.25, (j + 1) * 1.25, N_tr) + \
                np.random.randn(N_tr) * 0.05
            X_tr[ix] = np.c_[r * np.sin(t), r * np.cos(t)]

        for j in xrange(K):
            ix = range(N_ts * j, N_ts * (j + 1))
            r = np.linspace(2.5, 10.0, N_ts)
            t = np.linspace(j * 1.25, (j + 1) * 1.25, N_ts) + \
                np.random.randn(N_ts) * 0.05
            X_ts[ix] = np.c_[r * np.sin(t), r * np.cos(t)]

        return X_tr, X_ts

    def mnist(dataset="binary", dir="data/mnist"):
        from tensorflow.examples.tutorials.mnist import input_data

        mnist = input_data.read_data_sets("data/mnist/", one_hot=True)

        test_data = mnist.test.images
        train_data = mnist.train.images

        return train_data, test_data

    def graph(dataset="citeseer", dir="data/graphs/"):
        names = ["x", "tx", "allx", "graph"]
        objects = []
        for i in range(len(names)):
            objects.append(
                pkl.load(open(
                    dir + "ind.{}.{}".format(dataset, names[i]))
                )
            )
        x, tx, allx, graph = tuple(objects)
        test_idx_reorder = parse_index_file(
            dir + "ind.{}.test.index".format(dataset)
        )
        test_idx_range = np.sort(test_idx_reorder)

        if dataset == "citeseer":
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(
                min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        return adj, features

    if datagroup == "spiral":
        return spiral(**args)
    elif datagroup == "mnist":
        return mnist(**args)
    elif datagroup == "graph":
        return graph(**args)
    else:
        assert(False)


class Dataset:
    def __init__(self, data, batch_size=100, shuffle=True):
        self.data = np.copy(data)
        self.batch_size = batch_size

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


def sample_gumbel(shape, eps=1e-20):
    U = np.random.uniform(0, 1, shape)
    return - np.log(eps - np.log(U + eps))
