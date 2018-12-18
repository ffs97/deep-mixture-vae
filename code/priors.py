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
                "means", shape=(self.n_classes, self.dim), dtype=tf.float32,
                initializer=tf.initializers.random_normal
            )
            self.log_vars = tf.get_variable(
                "log_vars", shape=(self.n_classes, self.dim), dtype=tf.float32,
                initializer=tf.initializers.zeros
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

        means, log_vars = sess.run([self.means[c, :], self.log_vars[c, :]])
        samples = means + samples * np.exp(log_vars / 2.0)

        return samples

    def inverse_reparametrize(self, epsilon, parameters):
        assert("mean" in parameters and "log_var" in parameters)

        return parameters["mean"] + tf.exp(parameters["log_var"] / 2) * epsilon

    def get_cluster_weights(self, Z):
        Z = Z[:, None, :]

        means = self.means[None, :, :]
        log_vars = self.log_vars[None, :, :]

        probs = - (
            tf.reduce_sum(
                tf.square(Z - means) / tf.exp(log_vars), axis=-1
            ) / 2 + tf.reduce_sum(log_vars, axis=-1)
        )
        return tf.nn.softmax(probs)

    def kl_from_prior(self, parameters, eps=1e-20):
        assert(
            "cluster_sample" in parameters and
            "weights" in parameters and
            "log_var" in parameters and
            "mean" in parameters
        )

        mean = parameters["mean"]
        log_var = parameters["log_var"]

        weights = parameters["weights"]
        weights = tf.reshape(weights, (-1, self.n_classes))

        if parameters["cluster_sample"]:
            prior_mean = tf.matmul(weights, self.means)
            prior_log_var = tf.matmul(weights, self.log_vars)

            res = (
                prior_log_var - log_var - 1 +
                (
                    tf.exp(log_var) + tf.square(mean - prior_mean)
                ) / tf.exp(prior_log_var)
            )
            res = tf.reduce_mean(0.5 * tf.reduce_sum(res, axis=1))

        else:
            prior_means = self.means[None, :, :]
            prior_log_vars = self.log_vars[None, :, :]

            mean = mean[:, None, :]
            log_var = log_var[:, None, :]

            res = (
                prior_log_vars - log_var - 1 +
                (
                    tf.exp(log_var) + tf.square(mean - prior_means)
                ) / tf.exp(prior_log_vars)
            )
            res = tf.reduce_sum(res, axis=-1)
            res = tf.reduce_sum(res * weights, axis=-1)
            res = tf.reduce_mean(0.5 * res)

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
