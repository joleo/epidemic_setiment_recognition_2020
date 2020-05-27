# -*- coding: utf-8 -*-
"""
@Time ： 2020/3/15 14:52
@Auth ： joleo
@File ：utils.py
"""
from config import *
import numpy as np

'填充序列长度'
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])

def read_url_picture(df):
    i = 0
    len = df.shape[0]
    feature = np.empty((len, IMG_ROWS, IMG_COLS, 3), np.float16)  # 训练集五百张（500， 224， 224， 3）
    for row in df.itertuples():
        import requests as req
        from PIL import Image
        from io import BytesIO
        # print(row[5].replace('[','').replace(']','').split(','))
        # for url in row[5].replace('[','').replace(']','').split(','):
        # label = row[7]
        i = i + 1
        url = row[5].replace('[','').replace(']','').split(',')[0].replace('\'','')
        try:
            response = req.get(url)
            image = Image.open(BytesIO(response.content))
            image = image.resize((IMG_ROWS, IMG_COLS))
            # image.save(save_name + np.str(i) + '.jpg')
            # tmp = np.asarray(image)#.reshape(1, -1).T / 255
            feature[i,:,:,:]=image
        except: continue
    return feature

