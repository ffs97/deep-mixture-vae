import os
import models
import base_models
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
                    "Model to use [dmvae, vade, dmoe, dvmoe, vademoe]")
flags.DEFINE_string("model_name", "",
                    "Name of the model")
flags.DEFINE_string("dataset", "mnist",
                    "Dataset to use [mnist, spiral, cifar10]")

flags.DEFINE_integer("latent_dim", 10,
                     "Number of dimensions for latent variable Z")
flags.DEFINE_integer("output_dim", 1,
                     "Output dimension for regression variable for ME models")
flags.DEFINE_integer("n_classes", -1,
                     "Number of clusters or classes to use for ME models")

flags.DEFINE_boolean("classification", False,
                     "Whether the objective is classification or regression (ME models)")

flags.DEFINE_integer("n_epochs", 500,
                     "Number of epochs for training the model")
flags.DEFINE_integer("pretrain_epochs_vae", 200,
                     "Number of epochs for pretraining the vae model")
flags.DEFINE_integer("pretrain_epochs_gmm", 200,
                     "Number of epochs for pretraining the gmm model")

flags.DEFINE_boolean("plotting", True,
                     "Whether to generate sampling and regeneration plots")
flags.DEFINE_integer("plot_epochs", 100,
                     "Nummber of epochs before generating plots")

flags.DEFINE_integer("save_epochs", 10,
                     "Nummber of epochs before saving model")


def main(argv):
    dataset = FLAGS.dataset
    latent_dim = FLAGS.latent_dim
    output_dim = FLAGS.output_dim

    n_classes = FLAGS.n_classes

    model_str = FLAGS.model
    model_name = FLAGS.model_name

    plotting = FLAGS.plotting
    plot_epochs = FLAGS.plot_epochs

    save_epochs = FLAGS.save_epochs

    classification = FLAGS.classification

    moe = model_str[-3:] == "moe"

    n_epochs = FLAGS.n_epochs
    pretrain_epochs_vae = FLAGS.pretrain_epochs_vae
    pretrain_epochs_gmm = FLAGS.pretrain_epochs_gmm

    dataset = load_data(
        dataset, classification=classification, output_dim=output_dim
    )

    if model_name == "":
        model_name = model_str

    n_classes = dataset.n_classes
    if FLAGS.n_classes > 0:
        n_classes = FLAGS.n_classes

    if moe:
        n_experts = n_classes
        if not classification:
            n_experts = dataset.n_classes

        from includes.utils import MEDataset as Dataset

        if model_str not in ["dmoe", "vademoe", "dvmoe"]:
            raise NotImplementedError

        if model_str == "dmoe":
            model = models.DeepMoE(
                model_str, dataset.input_dim, output_dim, n_experts, classification,
                activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer
            )
            model.build_graph(
                [512, 256]
            )

            plotting = False

        elif model_str == "dvmoe":
            model = models.DVMoE(
                model_str, dataset.input_type, dataset.input_dim, latent_dim, output_dim, n_experts,
                classification, activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer
            ).build_graph(
                {"Z": [500, 500, 2000], "C": [256, 512]}, [2000, 500, 500]
            )

        elif model_str == "vademoe":
            model = models.VaDEMoE(
                model_str, dataset.input_type, dataset.input_dim, latent_dim, output_dim, n_experts,
                classification, activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer
            ).build_graph(
                [256, 256, 512], [512, 256]
            )

        test_data = (
            dataset.test_data, dataset.test_classes, dataset.test_labels
        )
        train_data = (
            dataset.train_data, dataset.train_classes, dataset.train_labels
        )

    else:
        n_clusters = n_classes

        from includes.utils import Dataset

        if model_str not in ["dmvae", "vade"]:
            raise NotImplementedError

        if model_str == "dmvae":
            model = base_models.DeepMixtureVAE(
                model_name, dataset.input_type, dataset.input_dim, latent_dim, n_clusters,
                activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer
            ).build_graph(
                # {"Z": [512, 256, 256], "C": [512, 256]}, [256, 256, 512]
                # {"Z": [512, 256, 256], "C": [512, 256]}, [256, 256, 512]
                {"Z": [2000, 500, 500], "C": [
                    2000, 500, 500]}, [500, 500, 2000]
            )
        elif model_str == "vade":
            model = base_models.VaDE(
                model_name, dataset.input_type, dataset.input_dim, latent_dim, n_clusters,
                activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer
            ).build_graph(
                {"Z": [512, 256, 256]}, [256, 256, 512]
            )

        dataset.train_data = np.concatenate(
            [dataset.train_data, dataset.test_data], axis=0
        )
        dataset.train_classes = np.concatenate(
            [dataset.train_classes, dataset.test_classes], axis=0
        )

        test_data = (dataset.test_data, dataset.test_classes)
        train_data = (dataset.train_data, dataset.train_classes)

    test_data = Dataset(test_data, batch_size=100)
    train_data = Dataset(train_data, batch_size=100)

    model.define_train_step(0.002, train_data.epoch_len * 10)

    if model_str in ["dmvae", "vade", "dvmoe", "vademoe"]:
        model.define_pretrain_step(0.005, train_data.epoch_len * 10)

    model.path = "saved_models/%s/%s" % (dataset.datagroup, model.name)
    for path in [model.path + "/" + x for x in ["model", "vae", "prior"]]:
        if not os.path.exists(path):
            os.makedirs(path)

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)

    if model_str in ["dvmoe", "vademoe"]:
        model.pretrain(
            sess, train_data, pretrain_epochs_vae
        )
    elif model_str in ["vade", "dmvae"]:
        model.pretrain(
            sess, train_data, pretrain_epochs_vae, pretrain_epochs_gmm
        )

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    saver = tf.train.Saver(var_list)
    ckpt_path = model.path + "/model/parameters.ckpt"

    try:
        saver.restore(sess, ckpt_path)
    except:
        print("Could not load trained model")

    with tqdm(range(n_epochs), postfix={"loss": "inf", "accy": "0.00%"}) as bar:
        accuracy = 0.0
        max_accuracy = 0.0

        for epoch in bar:
            if plotting and epoch % plot_epochs == 0:
                if dataset.sample_plot is not None:
                    dataset.sample_plot(model, sess)
                if dataset.regeneration_plot is not None:
                    dataset.regeneration_plot(model, test_data, sess)

            if epoch % save_epochs == 0:
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
        dataset.sample_plot(model, sess)
        dataset.regeneration_plot(model, test_data, sess)


if __name__ == "__main__":
    app.run(main)
