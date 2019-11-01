import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy.misc import imsave
import os
import shutil
from PIL import Image
import time
import random


img_height = 64
img_width = 64
img_layer = 3
img_size = img_height * img_width

pool_size = 50
ngf = 32
ndf = 64

n_z = 64

input_dim = 3


def build_encoder(batch_size, inputenc, name="encoder"):
    with tf.variable_scope(name):
        # Encode
        # x -> z_mean, z_sigma -> z
        # inputenc = tf.reshape(inputenc, [-1, inputenc.getshape().as_list()[0]])
        f1 = fc(inputenc, 256, scope='enc_fc1', activation_fn=tf.nn.relu)
        f2 = fc(f1, 128, scope='enc_fc2', activation_fn=tf.nn.relu)
        f3 = fc(f2, 64, scope='enc_fc3', activation_fn=tf.nn.relu)
        z_mu = fc(f3, n_z, scope='enc_fc4_mu', activation_fn=None)
        z_log_sigma_sq = fc(f3, n_z, scope='enc_fc4_sigma', activation_fn=None)
        eps = tf.random_normal(
            shape=tf.shape(z_log_sigma_sq),
            mean=0, stddev=1, dtype=tf.float32)
        z = z_mu + tf.sqrt(tf.exp(z_log_sigma_sq)) * eps

        return f2, z


def build_decoder(batch_size, inputenc, name="decoder"):
    with tf.variable_scope(name):
        g1 = fc(inputenc, 64, scope='dec_fc1', activation_fn=tf.nn.relu)
        g2 = fc(g1, 128, scope='dec_fc2', activation_fn=tf.nn.relu)
        g3 = fc(g2, 256, scope='dec_fc3', activation_fn=tf.nn.relu)
        x_hat = fc(g3, input_dim, scope='dec_fc4', activation_fn=tf.sigmoid)
        return x_hat


def build_network(batch_size, inputenc, name="whole_network"):
    with tf.variable_scope(name):

        # Encode
        # x -> z_mean, z_sigma -> z
        f1 = fc(inputenc, 256, scope='enc_fc1', activation_fn=tf.nn.relu)
        f2 = fc(f1, 128, scope='enc_fc2', activation_fn=tf.nn.relu)
        f3 = fc(f2, 64, scope='enc_fc3', activation_fn=tf.nn.relu)
        z_mu = fc(f3, n_z, scope='enc_fc4_mu', activation_fn=None)
        z_log_sigma_sq = fc(f3, n_z, scope='enc_fc4_sigma', activation_fn=None)
        eps = tf.random_normal(
            shape=tf.shape(z_log_sigma_sq),
            mean=0, stddev=1, dtype=tf.float32)
        z = z_mu + tf.sqrt(tf.exp(z_log_sigma_sq)) * eps

        g1 = fc(z, 64, scope='dec_fc1', activation_fn=tf.nn.relu)
        g2 = fc(g1, 128, scope='dec_fc2', activation_fn=tf.nn.relu)
        g3 = fc(g2, 256, scope='dec_fc3', activation_fn=tf.nn.relu)
        x_hat = fc(g3, input_dim, scope='dec_fc4', activation_fn=tf.sigmoid)

        return f2, g2, x_hat, z_mu, z_log_sigma_sq