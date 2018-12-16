import os
import models
import numpy as np
import tensorflow as tf
import matplotlib as mpl

from tqdm import tqdm

from absl import app
from absl import flags

from matplotlib import pyplot as plt
from matplotlib import gridspec as grid

import includes.visualization as visualization
from includes.utils import load_data, generate_regression_variable

mpl.rc_file_defaults()

tf.logging.set_verbosity(tf.logging.ERROR)


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model", "vademoe", "Model to use [dmvae, vade, dmoe, dvmoe, vademoe]"
)
flags.DEFINE_string(
    "dataset", "mnist", "Dataset to use [mnist, spiral]"
)

flags.DEFINE_integer(
    "latent_dim", 10, "Number of dimensions for latent variable Z"
)
flags.DEFINE_integer(
    "output_dim", 1, "Output dimension for regression variable for ME models"
)

flags.DEFINE_integer(
    "n_epochs", 500, "Number of epochs for training a model"
)

flags.DEFINE_boolean(
    "moe", True, "Whether to run the ME model"
)

flags.DEFINE_boolean(
    "plotting", True, "Whether to generate sampling and regeneration plots"
)
flags.DEFINE_integer(
    "plot_epochs", 100, "Nummber of epochs before generating plots"
)


def main(argv):
    dataset = FLAGS.dataset
    model_str = FLAGS.model
    latent_dim = FLAGS.latent_dim
    output_dim = FLAGS.output_dim

    plotting = FLAGS.plotting
    plot_epochs = FLAGS.plot_epochs

    n_epochs = FLAGS.n_epochs

    if dataset == "mnist":
        n_clusters = 10

        input_dim = 784
        input_type = "binary"

        sample_plot = visualization.mnist_sample_plot
        regeneration_plot = visualization.mnist_regeneration_plot

    elif dataset == "spiral":
        n_clusters = 5

        input_dim = 2
        input_type = "real"

        sample_plot = visualization.spiral_sample_plot
        regeneration_plot = visualization.spiral_regeneration_plot

    else:
        raise NotImplementedError

    train_data, test_data = load_data(dataset)

    if FLAGS.moe:
        from includes.utils import MEDataset as Dataset

        if model_str not in ["dmoe", "vademoe", "dvmoe"]:
            raise NotImplementedError

        train_data, test_data = generate_regression_variable(
            (train_data, test_data), output_dim, n_clusters
        )

        if model_str == "dmoe":
            model = models.DeepMoE(
                model_str, input_dim, output_dim, n_clusters,
                activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer
            )
            model.build_graph(
                [512, 256]
            )

            plotting = False

        elif model_str == "dvmoe":
            model = models.DVMoE(
                model_str, input_type, input_dim, latent_dim, output_dim, n_clusters,
                activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer
            )
            model.build_graph(
                {"Z": [256, 256, 512], "C": [256, 512]}, [512, 256]
            )

        elif model_str == "vademoe":
            model = models.VaDEMoE(
                model_str, input_type, input_dim, latent_dim, output_dim, 5,
                activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer
            )
            model.build_graph([512, 256], [256, 512])

    else:
        from includes.utils import Dataset

        if model_str not in ["dmvae", "vade"]:
            raise NotImplementedError

        (train_data, _), (test_data, _) = train_data, test_data

        if model_str == "dmvae":
            model = models.DiscreteMixtureVAE(
                model_str, input_type, input_dim, latent_dim, n_clusters,
                activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer
            )
            model.build_graph(
                {"Z": [256, 256, 512], "C": [256, 512]}, [512, 256]
            )
        elif model_str == "vade":
            model = models.VaDE(
                model_str, input_type, input_dim, latent_dim, n_clusters,
                activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer
            )
            model.build_graph([512, 256], [256, 512])

    train_data = Dataset(train_data, batch_size=200)
    test_data = Dataset(test_data, batch_size=200)

    model.define_train_step(0.002, train_data.epoch_len * 10)

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)

    if model_str in ["dvmoe", "vademoe"]:
        model.pretrain(sess, train_data, 100)

    with tqdm(range(n_epochs), postfix={"loss": "inf"}) as bar:
        for epoch in bar:
            if plotting and epoch % plot_epochs == 0 and epoch != 0:
                sample_plot(model, sess)
                regeneration_plot(model, test_data, sess)

            if FLAGS.moe:
                bar.set_postfix({
                    "loss": "%.4f" % model.train_op(sess, train_data),
                    "lsqe": "%.4f" % model.square_error(sess, test_data)
                })
            else:
                bar.set_postfix(
                    {"loss": "%.4f" % model.train_op(sess, train_data)}
                )

            if epoch % 5 == 0:
                model.testNow(sess, train_data)

    if plotting:
        sample_plot(model, sess)
        regeneration_plot(model, test_data, sess)


if __name__ == "__main__":
    app.run(main)
