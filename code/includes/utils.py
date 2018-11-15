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


def generate_regression_variable(data, output_dim, n_classes):
    (train_data, train_classes), (test_data, test_classes) = data

    input_dim = train_data.shape[1]

    biases = np.random.randn(output_dim, n_classes)
    weights = np.random.randn(output_dim, input_dim, n_classes)

    train_labels = np.swapaxes(np.matmul(train_data, weights), 0, 1) + biases
    train_labels = train_labels[range(len(train_data)), :, train_classes]
    # train_labels += np.random.standard_normal(train_labels.shape) * 0.01

    test_labels = np.swapaxes(np.matmul(test_data, weights), 0, 1) + biases
    test_labels = test_labels[range(len(test_data)), :, test_classes]

    return {
        "data": train_data, "labels": train_labels, "classes": train_classes
    }, {
        "data": test_data, "labels": test_labels, "classes": test_classes
    }


def load_data(datagroup, **args):
    def spiral(N_tr=5000, N_ts=1000, D=2, K=5):
        train_data = np.zeros((N_tr * K, D))
        test_data = np.zeros((N_ts * K, D))

        for j in range(K):
            ix = range(N_tr * j, N_tr * (j + 1))
            r = np.linspace(2.5, 10.0, N_tr)
            t = np.linspace(j * 1.25, (j + 1) * 1.25, N_tr) + \
                np.random.randn(N_tr) * 0.05
            train_data[ix] = np.c_[r * np.sin(t), r * np.cos(t)]

        for j in range(K):
            ix = range(N_ts * j, N_ts * (j + 1))
            r = np.linspace(2.5, 10.0, N_ts)
            t = np.linspace(j * 1.25, (j + 1) * 1.25, N_ts) + \
                np.random.randn(N_ts) * 0.05
            test_data[ix] = np.c_[r * np.sin(t), r * np.cos(t)]

        test_classes = np.arange(K).repeat(N_ts)
        train_classes = np.arange(K).repeat(N_tr)

        return (train_data, train_classes), (test_data, test_classes)

    def mnist(dir="data/mnist"):
        from tensorflow.examples.tutorials.mnist import input_data

        mnist = input_data.read_data_sets("data/mnist/", one_hot=False)

        test_data = mnist.test.images, mnist.test.labels
        train_data = mnist.train.images, mnist.train.labels

        return train_data, test_data

    if datagroup == "spiral":
        return spiral(**args)
    elif datagroup == "mnist":
        return mnist(**args)
    else:
        assert(False)


class MEDataset:
    def __init__(self, data, batch_size=100, shuffle=True):
        self.data = data["data"]
        self.labels = data["labels"]
        self.classes = data["classes"]

        self.shuffle = shuffle

        self.len = len(self.data)

        assert(len(self.labels) == self.len and len(self.classes) == self.len)

        self.batch_size = batch_size

        self.epoch_len = int(math.ceil(len(self.data) / batch_size))

    def get_batches(self):
        if self.shuffle:
            indices = np.random.permutation(len(self.data))

            self.data = self.data[indices]
            self.labels = self.labels[indices]
            self.classes = self.classes[indices]

        data_batch = list()
        labels_batch = list()
        classes_batch = list()

        count = 0
        for i in range(len(self.data)):
            data_batch.append(self.data[i])
            labels_batch.append(self.labels[i])
            classes_batch.append(self.classes[i])

            count += 1

            if count == self.batch_size:
                yield np.array(data_batch), np.array(labels_batch), np.array(classes_batch)

                data_batch = list()
                labels_batch = list()
                classes_batch = list()

                count = 0

        if count > 0:
            yield np.array(data_batch), np.array(labels_batch), np.array(classes_batch)

    def __len__(self):
        return self.epoch_len


class Dataset:
    def __init__(self, data, batch_size=100, shuffle=True):
        self.data = np.copy(data)
        self.batch_size = batch_size

        self.shuffle = shuffle

        self.data_dim = self.data.shape[1]

        self.epoch_len = int(math.ceil(len(data) / batch_size))

        if shuffle:
            np.random.shuffle(self.data)

    def get_batches(self):
        if self.shuffle:
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
