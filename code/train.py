import os
# import models
import vae_models
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

flags.DEFINE_string("model", "vademoe",
                    "Model to use [imsat, dmvae, vade, dmoe, dvmoe, vademoe]")
flags.DEFINE_string("dataset", "mnist",
                    "Dataset to use [mnist, spiral]")

flags.DEFINE_integer("latent_dim", 10,
                     "Number of dimensions for latent variable Z")
flags.DEFINE_integer("output_dim", 1,
                     "Output dimension for regression variable for ME models")
flags.DEFINE_integer("n_classes", -1,
                     "Number of clusters or classes to use for ME models")

flags.DEFINE_integer("n_epochs", 500,
                     "Number of epochs for training the model")
flags.DEFINE_integer("pretrain_epochs_vae", 200,
                     "Number of epochs for pretraining the vae model")
flags.DEFINE_integer("pretrain_epochs_gmm", 200,
                     "Number of epochs for pretraining the gmm model")
flags.DEFINE_integer("pretrain_epochs_dmvae", 200,
                     "Number of epochs for pretraining the dmvae model")

flags.DEFINE_boolean("plotting", True,
                     "Whether to generate sampling and regeneration plots")
flags.DEFINE_integer("plot_epochs", 100,
                     "Nummber of epochs before generating plots")


def main(argv):
    dataset = FLAGS.dataset
    model_str = FLAGS.model
    latent_dim = FLAGS.latent_dim
    output_dim = FLAGS.output_dim

    plotting = FLAGS.plotting
    plot_epochs = FLAGS.plot_epochs

    moe = model_str[-3:] == "moe"

    n_epochs = FLAGS.n_epochs
    pretrain_epochs_vae = FLAGS.pretrain_epochs_vae
    pretrain_epochs_gmm = FLAGS.pretrain_epochs_gmm
    pretrain_epochs_dmvae = FLAGS.pretrain_epochs_dmvae

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

    n_classes = n_clusters
    if FLAGS.n_classes > 0:
        n_classes = FLAGS.n_classes

    if moe:
        from includes.utils import MEDataset as Dataset

        if model_str not in ["dmoe", "vademoe", "dvmoe"]:
            raise NotImplementedError

        if model_str == "dmoe":
            model = models.DeepMoE(
                model_str, input_dim, output_dim, n_classes,
                activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer
            )
            model.build_graph(
                [512, 256]
            )
            plotting = False

        elif model_str == "dvmoe":
            model = models.DVMoE(
                model_str, input_type, input_dim, latent_dim, output_dim, n_classes,
                activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer
            ).build_graph(
                {"Z": [500, 500, 2000], "C": [256, 512]}, [2000, 500, 500]
            )

        elif model_str == "vademoe":
            model = models.VaDEMoE(
                model_str, input_type, input_dim, latent_dim, output_dim, n_classes,
                activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer
            ).build_graph(
                [256, 256, 512], [512, 256]
            )

        train_data, test_data = load_data(dataset)
        train_data, test_data = generate_regression_variable(
            (train_data, test_data), output_dim, n_clusters
        )

    else:
        from includes.utils import Dataset

        if model_str not in ["imsat", "dmvae", "vade"]:
            raise NotImplementedError

        if model_str == "dmvae":
            model = vae_models.DiscreteMixtureVAE(
                model_str, input_type, input_dim, latent_dim, n_classes,
                activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer
            ).build_graph(
                {"Z": [512, 256], "C": [512, 256]}, [256, 512]
            )
        elif model_str == "vade":
            model = vae_models.VaDE(
                model_str, input_type, input_dim, latent_dim, n_classes,
                activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer
            ).build_graph(
                {"Z": [512, 256, 256]}, [256, 256, 512]
            )
        elif model_str == "imsat":
            model = vae_models.IMSAT(
                model_str, input_type, input_dim, n_classes,
                activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer
            ).build_graph([1200, 1200])
            plotting = False


        (train_data, train_classes), (test_data, test_classes) = load_data(dataset)
        train_data = (
            np.concatenate([train_data, test_data], axis=0),
            np.concatenate([train_classes, test_classes], axis=0)
        )
        test_data = (test_data, test_classes)

    train_data = Dataset(train_data, batch_size=100)
    test_data = Dataset(test_data, batch_size=100)

    model.define_train_step(0.0005, train_data.epoch_len * 10)

    if model_str in ["dmvae", "vade", "dvmoe", "vademoe"]:
        model.define_pretrain_step(0.002, train_data.epoch_len * 10)

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)

    if model_str in ["dvmoe", "vademoe"]:
        model.pretrain(
            sess, train_data, pretrain_epochs_vae
        )
    elif model_str in ["vade", "dmvae"]:
        if model_str == "dmvae":
            model.pretrain(
                sess, train_data, pretrain_epochs_vae, pretrain_epochs_gmm, pretrain_epochs_dmvae
            )
        elif model_str == "vade":
            model.pretrain(
                sess, train_data, pretrain_epochs_vae, pretrain_epochs_gmm
            )

    with tqdm(range(n_epochs), postfix={"loss": "inf", "accy": "0.00%"}) as bar:
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        saver = tf.train.Saver(var_list)
        ckpt_path = "saved_models/%s/model/parameters.ckpt" % model.name

        try:
            saver.restore(sess, ckpt_path)
        except:
            print("Could not load trained model")

        accuracy = 0.0
        max_accuracy = 0.0

        for epoch in bar:
            # if plotting and epoch % plot_epochs == 0 and epoch != 0:
            if epoch % plot_epochs == 0:
                if plotting:
                    sample_plot(model, sess)
                    regeneration_plot(model, test_data, sess)

                accuracy = model.get_accuracy(sess, train_data)
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    saver.save(sess, ckpt_path)

            if moe:
                bar.set_postfix({
                    "loss": "%.4f" % model.train_op(sess, train_data),
                    "lsqe": "%.4f" % model.square_error(sess, test_data)
                })
            else:
                bar.set_postfix({
                    "loss": "%.4f" % model.train_op(sess, train_data),
                    "accy": "%.2f%%" % accuracy
                })

    if plotting:
        sample_plot(model, sess)
        regeneration_plot(model, test_data, sess)

if __name__ == "__main__":
    app.run(main)
