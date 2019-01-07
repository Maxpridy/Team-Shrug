# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
import config
import json
import os


def get_accuracy(dict, b, m, s, d):
    count = 0
    
    for i in range(len(b)):
        if dict['bcateid'][i].index(1) == b[i]:
            count += 1
    ret_b = count / len(b)

    count = 0
    for i in range(len(m)):
        if dict['mcateid'][i].index(1) == m[i]:
            count += 1
    ret_m = count / len(m)

    count = 0
    for i in range(len(s)):
        if dict['scateid'][i].index(1) == s[i]:
            count += 1
    len_s = (np.array(dict['scateid'])[:,0] != 1).sum() + 0.0000001
    ret_s = count / len_s

    count = 0
    for i in range(len(d)):
        if dict['dcateid'][i].index(1) == d[i]:
            count += 1
    len_d = (np.array(dict['dcateid'])[:,0] != 1).sum() + 0.0000001
    ret_d = count / len_d

    return [ret_b, ret_m, ret_s, ret_d]

def make_label_weights(dict, b_weight=1., m_weight=1.2, s_weight=1.3, d_weight=1.4):
    batch_labels = np.array(
        [np.argmax(dict[target], 1) for target in ['bcateid', 'mcateid', 'scateid', 'dcateid']]).astype('float32').T
    _label_weight = batch_labels.copy()
    
    _label_weight[:, 2:][_label_weight[:, 2:] != 0] = 1.
    _label_weight[:, 0] = b_weight
    _label_weight[:, 1] = m_weight
    _label_weight[:, 2] *= s_weight
    _label_weight[:, 3] *= d_weight

    return _label_weight