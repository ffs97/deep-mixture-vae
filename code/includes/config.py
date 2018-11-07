import os
import numpy as np


class Config:
    def __init__(self, dataset="mnist"):
        if dataset == "mnist":
            self.input_dim = 784

            self.latent_type = "binary"
            self.latent_dim = 50
            self.n_classes = 5

            self.latent_prior_weights = np.zeros(
                self.latent_dim, dtype=float
            )

            self.latent_temperature = 0.1
            self.latent_temperature_annealing = True
            self.latent_temperature_decay_steps = 10
            self.latent_temperature_decay_ratio = 0.3

            self.n_epochs = 500
            self.batch_size = 200

            self.regularizer = 1

            self.encoder_layer_sizes = [500, 500, 2000]
            self.decoder_layer_sizes = [2000, 500, 500]

            self.decay_steps = 10
            self.decay_rate = 0.9
            self.learning_rate = 0.002
            self.epsilon = 1e-04

        elif dataset == "spiral":
            self.input_dim = 2

            self.latent_type = "normal"
            self.latent_dim = 7

            self.n_epochs = 500
            self.batch_size = 200

            self.regularizer = 1

            self.encoder_layer_sizes = [500, 500, 2000]
            self.decoder_layer_sizes = [2000, 500, 500]

            self.decay_steps = 10
            self.decay_rate = 0.9
            self.learning_rate = 0.0001
            self.epsilon = 1e-04

        if not os.path.exists("models/" + dataset):
            os.makedirs("models/" + dataset)

        self.train_dir = "models/" + dataset
