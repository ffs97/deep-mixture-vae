import tensorflow as tf

def encoder_network(input, activation, initializer, reuse=None, cnn=True):

    with tf.variable_scope("encoder_network", reuse=tf.AUTO_REUSE):
        if cnn:
            X_flat = tf.reshape(input, (-1, 28, 28, 1))
            conv1 = tf.layers.conv2d(X_flat, filters=32, kernel_size=[5, 5], activation=tf.nn.relu, kernel_initializer=initializer)
            pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)   
            conv2 = tf.layers.conv2d(pool1, filters=32, kernel_size=[5, 5], activation=tf.nn.relu, kernel_initializer=initializer)
            pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)
            pool2_flat = tf.layers.flatten(pool2)
            hidden = tf.layers.dense(inputs=pool2_flat, units=500, activation=tf.nn.relu, kernel_initializer=initializer)

        else:

            hidden = tf.layers.dense(input, 500, activation=activation, kernel_initializer=initializer)
            hidden = tf.layers.dense(hidden, 500, activation=activation, kernel_initializer=initializer)

    return hidden

def Z_network(input, activation, initializer, latent_dim, reuse=None, cnn=True):
    with tf.variable_scope("z", reuse=tf.AUTO_REUSE):
        hidden_z = tf.layers.dense(input, 2000, activation=activation, kernel_initializer=initializer)
        mean = tf.layers.dense(hidden_z, latent_dim, activation=None, kernel_initializer=initializer)
        log_var = tf.layers.dense(hidden_z, latent_dim, activation=None, kernel_initializer=initializer)

    return mean, log_var

def C_network(input, activation, initializer, n_classes, reuse=None, cnn=True):
    with tf.variable_scope("c", reuse=tf.AUTO_REUSE):
        
        hidden_c = tf.layers.dense(input, 2000, activation=activation, kernel_initializer=initializer)
        hidden_c = tf.layers.dense(hidden_c, 500, activation=activation, kernel_initializer=initializer)
        hidden_c = tf.layers.dense(hidden_c, 250, activation=activation, kernel_initializer=initializer)
        logits = tf.layers.dense(hidden_c, n_classes, activation=None, kernel_initializer=initializer)
        cluster_probs = tf.nn.softmax(logits)

    return cluster_probs

def decoder_network(input, activation, initializer, input_dim, reuse=None, cnn=True):
    with tf.variable_scope("decoder_network", reuse=tf.AUTO_REUSE):
        hidden = tf.layers.dense(input, 2000, activation=activation, kernel_initializer=initializer)
        hidden = tf.layers.dense(hidden, 500, activation=activation, kernel_initializer=initializer)
        hidden = tf.layers.dense(input, 500, activation=activation, kernel_initializer=initializer)
        decoded_X = tf.layers.dense(hidden, input_dim, activation=None, kernel_initializer=initializer)

    return decoded_X

