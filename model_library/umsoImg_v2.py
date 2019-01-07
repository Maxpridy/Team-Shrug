# -*- coding:utf-8 -*-
from model_library.model_base import ModelBase
import config

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import batch_norm
from model_library.layers import *


class UmsoImg_v2(ModelBase):
    def __init__(self, input_features, labels=None, optional=None, is_training=True):
        super(UmsoImg_v2, self).__init__(labels, optional, is_training)
        self._product = input_features['product']
        self._img_feat = input_features['img_feat']

        num_filters = 128
        fc_hidden_size = 1024

        self.make_model(num_filters, fc_hidden_size)
        self.make_prediction()
        if self.is_training:
            self.make_loss()

    def make_loss(self):
        if not hasattr(self, 'logits_b') or not hasattr(self, 'logits_m') or not hasattr(self, 'logits_s') or not hasattr(self, 'logits_d'):
            raise Exception("make model first !!")
        losses1 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_b, logits=self.logits_b)
        self.loss1 = tf.reduce_mean(losses1, name="softmax_losses1")
        losses2 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_m, logits=self.logits_m)
        self.loss2 = tf.reduce_mean(losses2, name="softmax_losses2")
        losses3 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_s, logits=self.logits_s)
        self.loss3 = tf.reduce_mean(losses3, name="softmax_losses3")
        losses4 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_d, logits=self.logits_d)
        self.loss4 = tf.reduce_mean(losses4, name="softmax_losses4")

        with tf.name_scope("loss"):
            losses1 = tf.expand_dims(losses1, -1)
            losses2 = tf.expand_dims(losses2, -1)
            losses3 = tf.expand_dims(losses3, -1)
            losses4 = tf.expand_dims(losses4, -1)

            losses = tf.concat([losses1, losses2, losses3, losses4], axis=-1)
            losses *= self.label_weight
            self.total_loss = tf.reduce_mean(tf.reduce_mean(losses, -1), -1)

    def make_model(self, num_filters, fc_hidden_size):
        # define logits_b, logits_m, logits_s, logits_d
        word_feat = self.product_network(num_filters=num_filters, use_rnn=True)
        img_feat = tf.cast(self._img_feat, tf.float32)

        combine_feat = tf.concat([word_feat, img_feat], 1)

        with tf.name_scope("fc"):
            fc_out1_1 = FC(combine_feat, fc_hidden_size, activation='swish', use_bn=True, use_dropout=True, keep_prob=self.keep_prob, is_training=self.is_training)
            fc_out1_2 = FC(fc_out1_1, fc_hidden_size, activation='swish', use_bn=True, use_dropout=True, keep_prob=self.keep_prob, is_training=self.is_training)
            fc1_out = FC(fc_out1_2, config.big, use_bn=False, activation=None)
            self.logits_b = fc1_out

            fc_out2_1 = FC(fc_out1_2, fc_hidden_size, activation='swish', use_bn=True, use_dropout=True, keep_prob=self.keep_prob, is_training=self.is_training)
            fc_out2_2 = FC(fc_out2_1, fc_hidden_size, activation='swish', use_bn=True, use_dropout=True, keep_prob=self.keep_prob, is_training=self.is_training)
            fc2_out = FC(fc_out2_2, config.medium, use_bn=False, activation=None)
            self.logits_m = fc2_out

            fc_out3_1 = FC(fc_out2_2, fc_hidden_size, activation='swish', use_bn=True, use_dropout=True, keep_prob=self.keep_prob, is_training=self.is_training)
            fc_out3_2 = FC(fc_out3_1, fc_hidden_size, activation='swish', use_bn=True, use_dropout=True, keep_prob=self.keep_prob, is_training=self.is_training)
            fc3_out = FC(fc_out3_2, config.small, use_bn=False, activation=None)
            self.logits_s = fc3_out

            fc_out4_1 = FC(fc_out3_2, fc_hidden_size, activation='swish', use_bn=True, use_dropout=True, keep_prob=self.keep_prob, is_training=self.is_training)
            fc_out4_2 = FC(fc_out4_1, fc_hidden_size, activation='swish', use_bn=True, use_dropout=True, keep_prob=self.keep_prob, is_training=self.is_training)
            fc4_out = FC(fc_out4_2, config.detail, use_bn=False, activation=None)
            self.logits_d = fc4_out


    def product_network(self, lstm_hidden_size=256, num_filters=128, use_rnn=True):
        char_embedding = tf.get_variable('char_embedding', [config.character_size, config.embedding])
        embedded_product = tf.nn.embedding_lookup(char_embedding, self._product)

        if use_rnn:
            # Bi-LSTM Layer
            print("use rnn!")
            
            with tf.name_scope("Bi-lstm"):
                lstm_fw_cell = rnn.BasicLSTMCell(lstm_hidden_size)
                lstm_bw_cell = rnn.BasicLSTMCell(lstm_hidden_size)

                outputs_front, state_front = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, embedded_product, dtype=tf.float32)

                lstm_concat = tf.concat(outputs_front, axis=2)

                cnn_input = tf.expand_dims(lstm_concat, axis=-1) # (?, 90, 1024, 1)
            length_size = lstm_hidden_size * 2
            depth_size = 1
        else:
            cnn_input = tf.expand_dims(embedded_product, axis=-1)  # (?, 90, 32, 1)
            length_size = config.embedding
            depth_size = 1
            
            
        pooled_outputs = []
        filter_sizes = [3, 4, 5]
        for filter_size in filter_sizes:
            with tf.name_scope("conv-filter{0}".format(filter_size)):
                filter_shape = [filter_size, length_size, depth_size, num_filters]
                W = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1, dtype=tf.float32), name="W")
                conv = tf.nn.conv2d(
                    cnn_input,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                conv_bn = batch_norm(conv, is_training=self.is_training, trainable=True, updates_collections=None)
                conv_out = tf.nn.relu(conv_bn, name="conv_relu")

            with tf.name_scope("pool-filter{0}".format(filter_size)):
                avg_pooled = tf.nn.avg_pool(
                    conv_out,
                    ksize=[1, config.strmaxlen - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="avg_pool")

                max_pooled = tf.nn.max_pool(
                    conv_out,
                    ksize=[1, config.strmaxlen - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="max_pool")

                pooled_combine = tf.concat([avg_pooled, max_pooled], axis=3)

            pooled_outputs.append(pooled_combine)

        num_filters_total = num_filters * len(filter_sizes)

        pool = tf.concat(pooled_outputs, axis=3)
        word_feat = tf.reshape(pool, shape=[-1, num_filters_total * 2])

        return word_feat

