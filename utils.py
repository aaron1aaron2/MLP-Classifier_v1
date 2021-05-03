# encoding: utf-8
"""
@author: yen-nan ho
@contact: aaron1aaron2@gmail.com
"""
import os
import gzip
import json
import numpy as np

# 資料相關
def load_mnist(path, kind='train'):
    """程式碼來源: https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py"""
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

def save_json(newdata, path, indent=1):
    f = open(path, 'w', encoding='utf8')
    json.dump(newdata, f, indent=indent)
    f.close()

def save_filename(name, info_dt, path):
    text = name + '_' + '_'.join([f"{i}-{v}" for i,v in info_dt.items()])
    f = open(os.path.join(path, text), 'w', encoding='utf8')
    f.close()
