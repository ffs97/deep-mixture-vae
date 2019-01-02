import math
import numpy as np
import pickle as pkl
import scipy.sparse as sp

from includes import visualization
from sklearn.utils.linear_assignment_ import linear_assignment


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_gumbel(shape, eps=1e-20):
    U = np.random.uniform(0, 1, shape)
    return - np.log(eps - np.log(U + eps))


def get_clustering_accuracy(weights, classes):
    clusters = np.argmax(weights, axis=-1)

    n_classes = weights.shape[1]

    size = len(clusters)
    d = np.zeros((n_classes, n_classes), dtype=np.int32)

    for i in range(size):
        d[clusters[i], classes[i]] += 1

    ind = linear_assignment(d.max() - d)
    return sum([d[i, j] for i, j in ind]) / size


def generate_regression_variable(dataset, output_dim):
    n_experts = dataset.n_classes

    input_dim = dataset.train_data.shape[1]

    biases = np.random.randn(output_dim, n_experts)
    weights = np.random.randn(output_dim, input_dim, n_experts)

    train_labels = np.swapaxes(
        np.matmul(dataset.train_data, weights), 0, 1
    ) + biases
    train_labels = train_labels[range(
        len(dataset.train_data)), :, dataset.train_classes
    ]
    # train_labels += np.random.standard_normal(train_labels.shape) * 0.01

    test_labels = np.swapaxes(
        np.matmul(dataset.test_data, weights), 0, 1
    ) + biases
    test_labels = test_labels[
        range(len(dataset.test_data)), :, dataset.test_classes
    ]

    return train_labels, test_labels


def generate_classification_variables(dataset):
    n_classes = dataset.n_classes

    test_labels = np.zeros((len(dataset.test_classes), n_classes))
    test_labels[np.arange(0, len(dataset.test_classes)),
                dataset.test_classes] = 1

    train_labels = np.zeros((len(dataset.train_classes), n_classes))
    train_labels[np.arange(0, len(dataset.train_classes)),
                 dataset.train_classes] = 1

    return train_labels, test_labels


def load_data(datagroup, output_dim=1, classification=True, **args):
    def spiral(N_tr=5000, N_ts=1000, D=2, K=5):
        class SpiralDataset:
            pass

        dataset = SpiralDataset()

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

        dataset.datagroup = "spiral"

        dataset.test_data = test_data
        dataset.test_classes = test_classes

        dataset.train_data = train_data
        dataset.train_classes = train_classes

        dataset.n_classes = 5

        dataset.input_dim = 2
        dataset.input_type = "real"

        dataset.sample_plot = visualization.spiral_sample_plot
        dataset.regeneration_plot = visualization.spiral_regeneration_plot

        return dataset

    def mnist(dir="data/mnist"):
        from tensorflow.examples.tutorials.mnist import input_data

        class MNISTDataset:
            pass

        dataset = MNISTDataset()

        mnist = input_data.read_data_sets("data/mnist/", one_hot=False)

        dataset.datagroup = "mnist"

        dataset.test_data = mnist.test.images
        dataset.test_classes = mnist.test.labels

        dataset.train_data = mnist.train.images
        dataset.train_classes = mnist.train.labels

        dataset.n_classes = 10

        dataset.input_dim = 784
        dataset.input_type = "binary"

        dataset.sample_plot = visualization.mnist_sample_plot
        dataset.regeneration_plot = visualization.mnist_regeneration_plot

        return dataset

    def hhar(dir="data/hhar", filename="hhar.mat"):
        import scipy.io as scio

        class HHARDataset:
            pass

        dataset = HHARDataset()

        data = scio.loadmat(dir + "/" + filename)

        X = data["X"]
        Y = data["Y"] - 1

        p = np.random.permutation(X.shape[0])
        X = X[p]
        Y = Y[p]

        split = int(0.8 * len(X))
        train_data, test_data = X[:split], X[split:]
        train_classes, test_classes = Y[:split], Y[split:]

        dataset.datagroup = "hhar"

        dataset.test_data = test_data
        dataset.test_classes = test_classes

        dataset.train_data = train_data
        dataset.train_classes = train_classes

        dataset.n_classes = 6

        dataset.input_dim = 561
        dataset.input_type = "real"

        dataset.sample_plot = None
        dataset.regeneration_plot = None

        return dataset

    def cifar10(dir="data/cifar10"):
        from tensorflow.keras.datasets.cifar10 import load_data

        class Cifar10Dataset:
            pass

        dataset = Cifar10Dataset()

        (train_data, train_classes), (test_data, test_classes) = load_data()

        dataset.datagroup = "cifar10"

        dataset.test_classes = test_classes
        dataset.train_classes = train_classes

        dataset.test_data = np.reshape(test_data, (-1, 3072)) / 255
        dataset.train_data = np.reshape(train_data, (-1, 3072)) / 255

        dataset.n_classes = 10

        dataset.input_dim = 3072
        dataset.input_type = "binary"

        dataset.sample_plot = None
        dataset.regeneration_plot = visualization.cifar10_regeneration_plot

        return dataset

    def reuters(dir="data/reuters"):
        import os
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.feature_extraction.text import TfidfTransformer

        did_to_cat = {}
        cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']

        with open(os.path.join(dir, 'rcv1-v2.topics.qrels')) as fin:
            for line in fin.readlines():
                line = line.strip().split(' ')
                cat = line[0]
                did = int(line[1])
                if cat in cat_list:
                    did_to_cat[did] = did_to_cat.get(did, []) + [cat]

        # remove docs with multi labels
        keys = [i for i in did_to_cat.keys()]
        for did in keys:
            if len(did_to_cat[did]) > 1:
                del did_to_cat[did]

        dat_list = ['lyrl2004_tokens_test_pt0.dat',
                    'lyrl2004_tokens_test_pt1.dat',
                    'lyrl2004_tokens_test_pt2.dat',
                    'lyrl2004_tokens_test_pt3.dat',
                    'lyrl2004_tokens_train.dat']

        data = []
        target = []
        cat_to_cid = {'CCAT': 0, 'GCAT': 1, 'MCAT': 2, 'ECAT': 3}
        del did

        doc = ''
        for dat in dat_list:
            with open(os.path.join(dir, dat)) as fin:
                for line in fin.readlines():
                    if line.startswith('.I'):
                        if 'did' in locals():
                            if doc == '':
                                continue
                            if did in did_to_cat.keys():
                                data.append(doc)
                                target.append(cat_to_cid[did_to_cat[did][0]])
                        did = int(line.strip().split(' ')[1])
                        doc = ''
                    elif line.startswith('.W'):
                        assert doc == ''
                    else:
                        doc += line

        X = CountVectorizer(
            dtype=np.float64, max_features=2000).fit_transform(data)
        Y = np.asarray(target)

        X = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(X)
        X = np.asarray(X.todense()) * np.sqrt(X.shape[1])

        X = 2 * (X - X.min(0)) / (X.max(0) - X.min(0)) - 1

        p = np.random.permutation(X.shape[0])
        X = X[p]
        Y = Y[p]

        split = int(0.8 * len(X))
        train_data, test_data = X[:split], X[split:]
        train_classes, test_classes = Y[:split], Y[split:]

        class ReutersDataset:
            pass

        dataset = ReutersDataset()

        dataset.datagroup = "reuters"

        dataset.test_data = test_data
        dataset.test_classes = test_classes

        dataset.train_data = train_data
        dataset.train_classes = train_classes

        dataset.n_classes = 4

        dataset.input_dim = 2000
        dataset.input_type = "real"

        dataset.sample_plot = None
        dataset.regeneration_plot = None

        return dataset

    def reuters10k(dir="data/reuters", filename="reuters10k.mat"):
        import scipy.io as scio

        class Reuters10KDataset:
            pass

        dataset = Reuters10KDataset()

        data = scio.loadmat(dir + "/" + filename)

        X = data["X"]
        Y = data["Y"]

        X = 2 * (X - X.min(0)) / (X.max(0) - X.min(0)) - 1Z

        p = np.random.permutation(X.shape[0])
        X = X[p]
        Y = Y[p]

        split = int(0.8 * len(X))
        train_data, test_data = X[:split], X[split:]
        train_classes, test_classes = Y[:split], Y[split:]

        dataset.datagroup = "hhar"

        dataset.datagroup = "reuters"

        dataset.test_data = test_data
        dataset.test_classes = test_classes

        dataset.train_data = train_data
        dataset.train_classes = train_classes

        dataset.n_classes = 4

        dataset.input_dim = 2000
        dataset.input_type = "real"

        dataset.sample_plot = None
        dataset.regeneration_plot = None

        return dataset

    if datagroup == "spiral":
        dataset = spiral(**args)
    elif datagroup == "mnist":
        dataset = mnist(**args)
    elif datagroup == "hhar":
        dataset = hhar(**args)
    elif datagroup == "cifar10":
        dataset = cifar10(**args)
    elif datagroup == "reuters":
        dataset = reuters(**args)
    elif datagroup == "reuters10k":
        dataset = reuters10k(**args)
    else:
        raise NotImplementedError

    if classification:
        dataset.train_labels, dataset.test_labels = generate_classification_variables(
            dataset
        )
    else:
        dataset.train_labels, dataset.test_labels = generate_regression_variable(
            dataset, output_dim
        )

    return dataset


class MEDataset:
    def __init__(self, data, batch_size=100, shuffle=True):
        self.data, self.classes, self.labels = data

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
        data, classes = data

        self.data = np.copy(data)
        self.classes = np.copy(classes)

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.data_dim = self.data.shape[1]

        self.epoch_len = int(math.ceil(len(self.data) / batch_size))

        if shuffle:
            indices = np.random.permutation(len(self.data))

            self.data = self.data[indices]
            self.classes = self.classes[indices]

    def get_batches(self):
        if self.shuffle:
            indices = np.random.permutation(len(self.data))

            self.data = self.data[indices]
            self.classes = self.classes[indices]

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
