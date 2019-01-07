# -*- coding:utf-8 -*-

import argparse
import numpy as np
import tensorflow as tf

from dataset import *
import config
from model_library import baseline
from model_library import model_mgr
import util
import os


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='UmsoImg_v2_e31', help='path')

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


def run_test():
    args = parser.parse_args() 
    input_features, labels, optional = define_placeholder(use_img_feat=True)  # 입력값 정의

    model = model_mgr.get_model('umsoimg_v2')(input_features, labels, optional)

    _dataset = DatasetTest()
    
    sess = tf.InteractiveSession()
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=20)
    sess.run(init_op)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    try:
        if args.path == "UmsoImg_v2_e31":
            model_name = "UmsoImg_v2_e31.ckpt"
        else:
            model_name = args.path + ".ckpt"
    except:
        print("invalid filename")
    saver.restore(sess, os.path.join(config.result_model_dir, model_name))

    f = open("./what_the_test.tsv", "w")

    # epoch마다 학습을 수행합니다.
    for i, _dict in enumerate(_dataset.batch_loader(config.batch_size)):
        feed_dict = {
            input_features['product']: _dict['product'],
            input_features['img_feat']: _dict['img_feat'],
            optional['keep_prob']: 1.
        }
        step, _prediction = sess.run(
            [model.global_step, model.prediction], feed_dict)
        current_step = tf.train.global_step(sess, model.global_step)
        _b, _m, _s, _d = _prediction

        for pe, be, me, se, de in zip(_dict['pid'], _b, _m, _s, _d):
            f.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(pe.decode("utf-8"), be + 1, me + 1, se + 1, de + 1))
    print("test done!")


if __name__ == '__main__':
    run_test()
    

