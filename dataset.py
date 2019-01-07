# -*- coding:utf-8 -*-
import h5py
import numpy as np
import os
import config

class DataPreprocessing:
    def __init__(self):
        self.chosung = 19
        self.jungsung = 21
        self.jongsung = 27
        self.hangul_length = 67

    def decompose_as_one_hot2(self, in_char):
        one_hot = []
        if ord('가') <= in_char <= ord('힣'):
            base = in_char - 44032
            x = base // 21 // 28
            # y = base % 588 // 28
            y = base // 28 % 21
            z = base % 28

            if x >= self.chosung:
                # print('Unknown Exception: ', in_char, chr(in_char), x, y, z, zz)
                pass

            one_hot.append(x)
            one_hot.append(self.chosung + y)
            if z > 0:
                one_hot.append(self.chosung + self.jungsung + (z - 1))
            return one_hot
        else:
            if in_char < 128:
                return [self.hangul_length + in_char]
            elif ord('ㄱ') <= in_char <= ord('ㅣ'):
                return [self.hangul_length + 128 + (in_char - 12593)]
            else:
                # print('Unhandled character:', chr(in_char), in_char)
                return []

    def decompose_str_as_one_hot1(self, string):
        tmp_list = []
        for x in string:
            tmp_list.extend(self.decompose_as_one_hot2(ord(x)))
        return tmp_list

    def preprocessing(self, data, max_length=config.strmaxlen):
        umso_convert_list = []
        for ele in data:
            umso_convert_list.append(self.decompose_str_as_one_hot1(ele))
        padding_umso = np.zeros((len(data), max_length), dtype=np.int32)
        for i, sentence in enumerate(umso_convert_list):
            length = len(sentence)
            if length >= max_length:
                length = max_length
                padding_umso[i, :length] = np.array(sentence)[:length]
            else:
                padding_umso[i, :length] = np.array(sentence)

        return padding_umso


class Dataset(object):
    def __init__(self):
        self.data_processer = DataPreprocessing()

    def __len__(self):
        return self.length

    def make_onehot(self, data, classes):
        result = []
        for e in data:
            _list = [0.0 for _ in range(classes)]
            if e != -1:
                _list[e - 1] = 1.0
            else:
                _list[0] = 1.0
            result.append(_list)
        return result

    def batch_loader(self, n):
        raise Exception("no implements")


class DatasetAll(Dataset):
    def __init__(self, target=None):
        # target는 학습에 사용할 DB 번호 리스트
        super(DatasetAll, self).__init__()  # python 2.7에도 작동
        self.db_range = list(range(1, 10)) if target is None else target
        self.db_paths = [os.path.join(config.train_db_dir, 'train.chunk.0{}'.format(i)) for i in self.db_range]

    def batch_loader(self, n):
        for idx, db_path in enumerate(self.db_paths):
            self.now_dataset = self.db_range[idx] 
            data = h5py.File(db_path, 'r')['train']
            print('DB #{} loaded'.format(self.db_range[idx]))
            self.length = len(data['pid'])
            for i in range(0, self.length, n):
                last = min(i + n, self.length)
                yield {'pid': data['pid'][i:last],
                       'product': self.data_processer.preprocessing(list(map(lambda x: str(x, "utf-8"), data['product'][i:last]))),
                       'img_feat': data['img_feat'][i:last],
                       'price': data['price'][i:last],
                       'bcateid': self.make_onehot(data['bcateid'][i:last], config.big),
                       'mcateid': self.make_onehot(data['mcateid'][i:last], config.medium),
                       'scateid': self.make_onehot(data['scateid'][i:last], config.small),
                       'dcateid': self.make_onehot(data['dcateid'][i:last], config.detail)
                       }
            del data 


class DatasetTest(Dataset):
    def __init__(self):
        super(DatasetTest, self).__init__()  # python 2.7에도 작동
        self.db_range = list(range(1, 3)) # 1, 2
        self.db_paths = [os.path.join(config.test_db_dir, 'test.chunk.0{}'.format(i)) for i in self.db_range]

    def batch_loader(self, n):
        for idx, db_path in enumerate(self.db_paths):
            data = h5py.File(db_path, 'r')['test']
            self.length = len(data['pid'])
            for i in range(0, self.length, n):
                last = min(i + n, self.length)
                yield {'pid': data['pid'][i:last],
                       'product': self.data_processer.preprocessing(list(map(lambda x: str(x, "utf-8"), data['product'][i:last]))),
                       'img_feat': data['img_feat'][i:last],
                       'price': data['price'][i:last],
                       'bcateid': self.make_onehot(data['bcateid'][i:last], config.big),
                       'mcateid': self.make_onehot(data['mcateid'][i:last], config.medium),
                       'scateid': self.make_onehot(data['scateid'][i:last], config.small),
                       'dcateid': self.make_onehot(data['dcateid'][i:last], config.detail)
                       }
            del data


# class DatasetDev(Dataset):
#     def __init__(self):
#         # target는 학습에 사용할 DB 번호 리스트
#         super(DatasetDev, self).__init__()  # python 2.7에도 작동
#         self.db_paths = [os.path.join(config.dev_db_dir, 'dev.chunk.01')]

#     def batch_loader(self, n):
#         for idx, db_path in enumerate(self.db_paths):
#             data = h5py.File(db_path, 'r')['dev']
#             self.length = len(data['pid'])
#             for i in range(0, self.length, n):
#                 last = min(i + n, self.length)
#                 yield {'pid': data['pid'][i:last],
#                        'product': self.data_processer.preprocessing(list(map(lambda x: str(x, "utf-8"), data['product'][i:last]))),
#                        'img_feat': data['img_feat'][i:last],
#                        'price': data['price'][i:last],
#                        'bcateid': self.make_onehot(data['bcateid'][i:last], config.big),
#                        'mcateid': self.make_onehot(data['mcateid'][i:last], config.medium),
#                        'scateid': self.make_onehot(data['scateid'][i:last], config.small),
#                        'dcateid': self.make_onehot(data['dcateid'][i:last], config.detail)
#                        }
#             del data 