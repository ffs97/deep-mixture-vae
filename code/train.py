import os
import moe
import numpy as np
import tensorflow as tf
import matplotlib as mpl

from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import gridspec as grid

from includes.config import Config
from includes.utils import load_data, Dataset

mpl.rc_file_defaults()

tf.logging.set_verbosity(tf.logging.ERROR)


folders = ["models", "plots"]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)


def regeneration_plot():
    gs = grid.GridSpec(1, 2)

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    def reshape(images):
        images = images.reshape((10, 10, 28, 28))
        images = np.concatenate(np.split(images, 10, axis=0), axis=3)
        images = np.concatenate(np.split(images, 10, axis=1), axis=2)
        images = np.squeeze(images)

        return images

    eps = moem.vae.sample_reparametrization_variables(100, feed=True)

    orig_X = test_data.data[:100]
    recn_X = sess.run(
        moem.vae.reconstructed_X, feed_dict={
            moem.vae.X: orig_X,
            moem.vae.epsilon: eps["Z"],
            moem.vae.cluster: eps["C"]
        }
    )

    ax1.imshow(reshape(orig_X), cmap="Greys_r")
    ax2.imshow(reshape(recn_X), cmap="Greys_r")

    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)

    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig("plots/regenerated.png", transparent=True)


def sample_plot():
    figure = np.zeros((280, 280))

    for i in range(10):
        for j in range(10):
            kwargs = {
                "Z": {
                    "session": sess,
                    "c": i
                }
            }
            eps = moem.vae.sample_generative_feed(1, **kwargs)
            out = sess.run(moem.vae.reconstructed_X,
                           feed_dict={moem.vae.Z: eps["Z"]})

            figure[i * 28: (i + 1) * 28, j * 28: (j + 1) *
                   28] = out.reshape((28, 28)) * 255

    ax = plt.axes()
    ax.imshow(figure, cmap="Greys_r")

    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.savefig("plots/sampled.png", transparent=True)

    plt.close()


datagroup = "mnist"
dataset = "binary"

train_data, test_data = load_data(datagroup, dataset=dataset, output_dim=2)

train_data = Dataset(train_data, batch_size=200)
test_data = Dataset(test_data, batch_size=200)

model = "dmoe"

if model == "vmoe":
    moem = moe.VMoE("vmoe", 784, 10, 2, 10, activation=tf.nn.relu,
                    initializer=tf.contrib.layers.xavier_initializer)
    moem.build_graph({"Z": [512, 256], "C": [512, 256]}, [256, 512])

elif model == "dmoe":
    moem = moe.DMoE("gmmoe", 784, 2, 10, activation=tf.nn.relu,
                    initializer=tf.contrib.layers.xavier_initializer)
    moem.build_graph([512, 256])

moem.define_train_step(0.002, train_data.epoch_len * 10)

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

with tqdm(range(1000), postfix={"loss": "inf", "lsqe": "inf"}) as bar:
    for epoch in bar:
        if model == "vmoe" and epoch % 10 == 0:
            sample_plot()
            regeneration_plot()

        bar.set_postfix({
            "loss": "%.4f" % moem.train_op(sess, train_data),
            "lsqe": "%.4f" % moem.square_error(sess, test_data)
        })

if model == "vmoe":
    sample_plot()
    regeneration_plot()

print moem.square_error(sess, test_data)
