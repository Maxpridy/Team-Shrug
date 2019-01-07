# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf

from dataset import *
import config
from model_library import baseline
from model_library import model_mgr
import util
import os

def define_placeholder(use_img_feat=False):
    ## 입력값 정의 ##
    # input
    input_features = dict()
    input_features['product'] = tf.placeholder(tf.int32, [None, config.strmaxlen], name="product")
    if use_img_feat:
        input_features['img_feat'] = tf.placeholder(tf.int32, [None, config.img_feature_length], name="img_feat")

    # output
    labels = dict()
    labels['b'] = tf.placeholder(tf.int32, [None, config.big], name="y_b")
    labels['m'] = tf.placeholder(tf.int32, [None, config.medium], name="y_m")
    labels['s'] = tf.placeholder(tf.int32, [None, config.small], name="y_s")
    labels['d'] = tf.placeholder(tf.int32, [None, config.detail], name="y_d")

    # optional
    optional = dict()
    optional['label_weight'] = tf.placeholder(tf.float32, [None, 4], name='label_weight')
    optional['keep_prob'] = tf.placeholder(tf.float32, name='keep_prob')

    return input_features, labels, optional


def run_train():
    input_features, labels, optional = define_placeholder(use_img_feat=True)  # 입력값 정의

    # model = baseline.Baseline(input_features, labels, optional)  # 모델 구축
    model = model_mgr.get_model('umsoimg_v2')(input_features, labels, optional)

    # 최적화 알고리즘 정의
    optimizer = tf.train.AdamOptimizer(config.learning_rate)
    grads_and_vars = optimizer.compute_gradients(model.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=model.global_step, name="train_op")

    _dataset = DatasetAll(target=None)

    sess = tf.InteractiveSession()
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=20)
    sess.run(init_op)

    for i in range(config.epochs):
        for j, _dict in enumerate(_dataset.batch_loader(config.batch_size)):
            _label_weight = util.make_label_weights(_dict, b_weight=1., m_weight=1.2, s_weight=1.3,
                                                    d_weight=1.4)  # label 별 weight 생성
            feed_dict = {
                input_features['product']: _dict['product'],
                input_features['img_feat']: _dict['img_feat'],
                labels['b']: _dict['bcateid'],
                labels['m']: _dict['mcateid'],
                labels['s']: _dict['scateid'],
                labels['d']: _dict['dcateid'],
                optional['label_weight']: _label_weight,
                optional['keep_prob']: 0.7
            }
            _, step, _loss, _prediction = sess.run([train_op, model.global_step, model.loss, model.prediction], feed_dict)
            _b, _m, _s, _d = _prediction

            if j % 100 == 0:
                result = util.get_accuracy(_dict, _b, _m, _s, _d)
                print("epoch : {} [t:{}], global_step: {}, Acc: {}, loss: {}".format(
                    i + 1, _dataset.now_dataset, step,
                    [str(result[0])[:6], str(result[1])[:6], str(result[2])[:6], str(result[3])[:6]],
                    str(_loss)[:6]))

        model_name = '{}_e{}.ckpt'.format(model.__class__.__name__, str(i) if i >= 10 else ('0' + str(i)))
        print("save model... {}".format(model_name))
        # save_path = saver.save(sess, "../tmp/{}".format(model_name))
        save_path = saver.save(sess, os.path.join(config.result_model_dir, model_name))


if __name__ == '__main__':
    run_train()


