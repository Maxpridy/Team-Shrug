#-*- coding:utf-8 -*-
from model_library.model_base import ModelBase
import config

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import batch_norm
from model_library.layers import *

class Baseline(ModelBase):
    def __init__(self, input_features, labels=None, optional=None, is_training=True):
        super(Baseline, self).__init__(labels, optional, is_training)
        self._product = input_features['product']

        _lstm_hidden_size = 256
        num_filters = 128
        fc_hidden_size = 1024

        self.make_model(_lstm_hidden_size, num_filters, fc_hidden_size)
        self.make_prediction()
        if self.is_training:
            self.make_loss()

    def make_model(self, _lstm_hidden_size, num_filters, fc_hidden_size):
        # define logits_b, logits_m, logits_s, logits_d

        char_embedding = tf.get_variable('char_embedding', [config.character_size, config.embedding])
        embedded_product = tf.nn.embedding_lookup(char_embedding, self._product)

        # Bi-LSTM Layer
        with tf.name_scope("Bi-lstm"):
            lstm_fw_cell = rnn.BasicLSTMCell(_lstm_hidden_size)
            lstm_bw_cell = rnn.BasicLSTMCell(_lstm_hidden_size)

            outputs_front, state_front = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell, lstm_bw_cell, embedded_product, dtype=tf.float32)

            lstm_concat = tf.concat(outputs_front, axis=2)

            lstm_out = tf.expand_dims(lstm_concat, axis=-1)

        pooled_outputs = []

        filter_sizes = [3, 4, 5]
        for filter_size in filter_sizes:
            with tf.name_scope("conv-filter{0}".format(filter_size)):
                filter_shape = [filter_size, _lstm_hidden_size * 2, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1, dtype=tf.float32), name="W")
                b = tf.Variable(tf.constant(value=0.1, shape=[num_filters], dtype=tf.float32), name="b")
                conv = tf.nn.conv2d(
                    lstm_out,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                conv = tf.nn.bias_add(conv, b)

                conv_bn = batch_norm(conv, is_training=self.is_training, trainable=True, updates_collections=None)
                conv_out = tf.nn.relu(conv_bn, name="relu_front")

            with tf.name_scope("pool-filter{0}".format(filter_size)):
                avg_pooled = tf.nn.avg_pool(
                    conv_out,
                    ksize=[1, config.strmaxlen - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool")

                max_pooled = tf.nn.max_pool(
                    conv_out,
                    ksize=[1, config.strmaxlen - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool")

                pooled_combine = tf.concat([avg_pooled, max_pooled], axis=3)

            pooled_outputs.append(pooled_combine)

        num_filters_total = num_filters * len(filter_sizes)

        pool = tf.concat(pooled_outputs, axis=3)
        pool_flat = tf.reshape(pool, shape=[-1, num_filters_total * 2])

        with tf.name_scope("fc"):
            fc_out1 = FC(pool_flat, fc_hidden_size, activation='relu', use_bn=True, is_training=self.is_training)
            fc1_out = FC(fc_out1, config.big, use_bn=False, activation=None)
            self.logits_b = fc1_out

            fc_out2 = FC(fc_out1, fc_hidden_size, activation='relu', use_bn=True, is_training=self.is_training)
            fc2_out = FC(fc_out2, config.medium, use_bn=False, activation=None)
            self.logits_m = fc2_out

            fc_out3 = FC(fc_out2, fc_hidden_size, activation='relu', use_bn=True, is_training=self.is_training)
            fc3_out = FC(fc_out3, config.small, use_bn=False, activation=None)
            self.logits_s = fc3_out

            fc_out4 = FC(fc_out3, fc_hidden_size, activation='relu', use_bn=True, is_training=self.is_training)
            fc4_out = FC(fc_out4, config.detail, use_bn=False, activation=None)

            self.logits_d = fc4_out