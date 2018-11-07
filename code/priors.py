import qupa
import qupa.pcd as pcd
import numpy as np
import tensorflow as tf

from includes.utils import sample_gumbel


class LatentVariable:
    def kl_from_prior(self, **kwargs):
        raise NotImplementedError

    def sample_reparametrization_variable(self, **kwargs):
        raise NotImplementedError

    def sample_generative_feed(self, **kwargs):
        raise NotImplementedError

    def inverse_reparametrize(self, **kwargs):
        raise NotImplementedError


class NormalFactorial(LatentVariable):
    def __init__(self, name, dim):
        self.name = name

        self.dim = dim

    def sample_reparametrization_variable(self, n):
        return np.random.randn(n, self.dim)

    def sample_generative_feed(self, n, **kwargs):
        return np.random.randn(n, self.dim)

    def inverse_reparametrize(self, epsilon, parameters):
        assert("mean" in parameters and "log_var" in parameters)

        return parameters["mean"] + tf.exp(parameters["log_var"] / 2) * epsilon

    def kl_from_prior(self, parameters, eps=1e-20):
        assert("mean" in parameters and "log_var" in parameters)

        mean = parameters["mean"]
        log_var = parameters["log_var"]

        res = tf.exp(log_var) + tf.square(mean) - 1. - log_var
        res = tf.reduce_mean(0.5 * tf.reduce_sum(res, axis=1))

        return res


class NormalMixtureFactorial(LatentVariable):
    def __init__(self, name, dim, n_classes):
        self.name = name

        self.dim = dim
        self.n_classes = n_classes

        with tf.variable_scope(self.name) as _:
            self.means = tf.get_variable(
                "means", shape=(self.n_classes, self.dim), dtype=tf.float32
            )

    def sample_reparametrization_variable(self, n):
        return np.random.randn(n, self.dim)

    def sample_generative_feed(self, n, **kwargs):
        assert("session" in kwargs)

        samples = np.random.randn(n, self.dim)

        sess = kwargs["session"]
        if "c" not in kwargs:
            c = np.random.randint(0, 10, n, dtype=np.int32)
        else:
            c = kwargs["c"]

        samples = sess.run(self.means)[c, :] + samples

        return samples

    def inverse_reparametrize(self, epsilon, parameters):
        assert("mean" in parameters and "log_var" in parameters)

        return parameters["mean"] + tf.exp(parameters["log_var"] / 2) * epsilon

    def kl_from_prior(self, parameters, eps=1e-20):
        assert(
            "weights" in parameters and
            "log_var" in parameters and
            "mean" in parameters
        )

        mean = parameters["mean"]
        log_var = parameters["log_var"]

        weights = parameters["weights"]
        weights = tf.reshape(weights, (-1, self.n_classes))

        prior_mean = tf.matmul(weights, self.means)

        res = tf.exp(log_var) - log_var + tf.square(mean - prior_mean) - 1
        res = tf.reduce_mean(0.5 * tf.reduce_sum(res, axis=1))

        return res


class DiscreteFactorial(LatentVariable):
    def __init__(self, name, dim, n_classes):
        self.name = name

        self.dim = dim
        self.n_classes = n_classes

    def sample_reparametrization_variable(self, n):
        return sample_gumbel((n, self.dim, self.n_classes))

    def sample_generative_feed(self, n, **kwargs):
        samples = sample_gumbel((n, self.dim, self.n_classes))
        samples = np.reshape(samples, (-1, self.n_classes))
        samples = np.asarray(np.equal(
            samples, np.max(samples, 1, keepdims=True)
        ), dtype=samples.dtype)
        samples = np.reshape(samples, (-1, self.dim, self.n_classes))

        return samples

    def inverse_reparametrize(self, epsilon, parameters):
        assert("logits" in parameters and "temperature" in parameters)

        logits = parameters["logits"]
        logits = tf.reshape(logits, (-1, self.n_classes))

        res = tf.reshape(epsilon, (-1, self.n_classes))
        res = (logits + res) / parameters["temperature"]
        res = tf.nn.softmax(res)
        res = tf.reshape(res, (-1, self.dim, self.n_classes))

        return res

    def kl_from_prior(self, parameters, eps=1e-20):
        assert("logits" in parameters)

        logits = tf.reshape(parameters["logits"], (-1, self.n_classes))
        q_z = tf.nn.softmax(logits)

        res = tf.reshape(
            q_z * (tf.log(q_z + eps) - tf.log(1.0 / self.n_classes)),
            (-1, self.dim * self.n_classes)
        )
        res = tf.reduce_mean(tf.reduce_sum(res, axis=1))

        return res
