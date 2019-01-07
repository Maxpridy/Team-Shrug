#-*- coding:utf-8 -*-

import config
import tensorflow as tf

class ModelBase(object):
    def __init__(self, labels, optional, is_training=True):
        self.is_training = is_training
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")

        if 'keep_prob' in optional:
            self.keep_prob = optional['keep_prob']

        if is_training:
            self.y_b = labels['b']
            self.y_m = labels['m']
            self.y_s = labels['s']
            self.y_d = labels['d']
            self.label_weight = optional['label_weight']

    def make_model(self):
        # define logits_b, logits_m, logits_s, logits_d
        #   self.logits_b, m, s, d
        raise Exception("implement model_library")

    def make_prediction(self):
        if not hasattr(self, 'logits_b') or not hasattr(self, 'logits_m') or not hasattr(self, 'logits_s') or not hasattr(self, 'logits_d'):
            raise Exception("make model first !!")
        self.pred_b = tf.argmax(self.logits_b, 1, name="predict_b")
        self.pred_m = tf.argmax(self.logits_m, 1, name="predict_m")
        self.pred_s = tf.argmax(self.logits_s, 1, name="predict_s")
        self.pred_d = tf.argmax(self.logits_d, 1, name="predict_d")

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
            # loss_ab = loss1 + loss2 * 1.2
            # loss_abc = loss1 + loss2 * 1.2 + loss3 * 1.3

            losses1 = tf.expand_dims(losses1, -1)
            losses2 = tf.expand_dims(losses2, -1)
            losses3 = tf.expand_dims(losses3, -1)
            losses4 = tf.expand_dims(losses4, -1)
            # print(loss1.get_shape())
            losses = tf.concat([losses1, losses2, losses3, losses4], axis=-1)
            # print(losses.get_shape())
            # print(label_weight.get_shape())
            losses *= self.label_weight
            self.total_loss = tf.reduce_mean(tf.reduce_mean(losses, -1), -1)
            # loss_abcd = loss1 + loss2*1.2 + loss3*1.3 + loss4*1.4

    @property
    def loss(self):
        if not hasattr(self, 'total_loss'):
            raise Exception("make loss first !!")
        return self.total_loss

    @property
    def losses(self):
        if not hasattr(self, 'total_loss'):
            raise Exception("make loss first !!")
        return [self.loss1, self.loss2, self.loss3, self.loss4]

    @property
    def prediction(self):
        if not hasattr(self, 'pred_b') or not hasattr(self, 'pred_m') or not hasattr(self, 'pred_s') or not hasattr(self, 'pred_d'):
            raise Exception("make model first !!")
        return [self.pred_b, self.pred_m, self.pred_s, self.pred_d]
