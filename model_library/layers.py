import tensorflow as tf
from tensorflow.contrib.layers import batch_norm


def FC(x, output_dims, use_bn=False, use_dropout=False, keep_prob=None, activation='relu', is_training=True):
    W = tf.Variable(
        tf.truncated_normal(shape=[x.get_shape()[-1].value, output_dims], stddev=0.1, dtype=tf.float32))
    fc = tf.matmul(x, W)
    if not use_bn:
        b = tf.Variable(tf.constant(value=0.1, shape=[output_dims], dtype=tf.float32))
        fc = tf.nn.bias_add(fc, b)
    if use_bn:
        fc = batch_norm(fc, is_training=is_training, trainable=True, updates_collections=None)

    if activation is not None:
        # relu, swish, relu6, selu, elu, sigmoid, softmax ...
        fc = eval('tf.nn.{}(fc)'.format(activation))
    if use_dropout:
        fc = tf.nn.dropout(fc, keep_prob)
    return fc
