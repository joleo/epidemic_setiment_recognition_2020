# -*- coding: utf-8 -*-
"""
@Time ： 2020/3/15 14:08
@Auth ： joleo
@File ：mul_bert_with_pic.py
"""

from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.layers import Flatten, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing import image
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import concatenate
import pandas
from keras.optimizers import SGD,Adam,RMSprop
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
"""
创建alexnet网络模型
"""
# 分类数量
# nb_classes = 3
def alex_net(w_path=None):
    # 输入图像为（3， 224， 224）
    input_shape = (224, 224, 3)
    # 输入层
    inputs = Input(shape=input_shape, name='input')

    # 第一层：两个卷积操作和两个pooling操作

    conv1_1 = Convolution2D(48, (11, 11), strides=(4, 4), activation='relu', name='conv1_1')(inputs)
    conv1_2 = Convolution2D(48, (11, 11), strides=(4, 4), activation='relu', name='conv1_2')(inputs)

    pool1_1 = MaxPooling2D((3, 3), strides=(2, 2), name='pool1_1')(conv1_1)
    pool1_2 = MaxPooling2D((3, 3), strides=(2, 2), name='pool1_2')(conv1_2)

    # layer2:两个卷积，将前边的池化得到的数据，进行卷积，再继续池化

    conv2_1 = Convolution2D(128, (5, 5), activation='relu', padding='same')(pool1_1)
    conv2_2 = Convolution2D(128, (5, 5), activation='relu', padding='same')(pool1_2)

    pool2_1 = MaxPooling2D((3, 3), strides=(2, 2), name='pool2_1')(conv2_1)
    pool2_2 = MaxPooling2D((3, 3), strides=(2, 2), name='pool2_2')(conv2_2)

    # merge合并层：第二层进入第三层，将数据混合合并
    merge1 = concatenate([pool2_2, pool2_1], axis=1)

    # layer3:两个卷积操作

    conv3_1 = Convolution2D(192, (3, 3), activation='relu', name='conv3_1', padding='same')(merge1)
    conv3_2 = Convolution2D(193, (3, 3), activation='relu', name='conv3_2', padding='same')(merge1)

    # latyer4:两个卷积操作
    conv4_1 = Convolution2D(192, (3, 3), activation='relu', name='conv4_1', padding='same')(conv3_1)
    conv4_2 = Convolution2D(192, (3, 3), activation='relu', name='conv4_2', padding='same')(conv3_2)

    # layer5:两个卷积操作和两个pooling操作
    conv5_1 = Convolution2D(128, (3, 3), activation='relu', name='conv5_1', padding='same')(conv4_1)
    conv5_2 = Convolution2D(128, (3, 3), activation='relu', name='conv5_2', padding='same')(conv4_2)

    pool5_1 = MaxPooling2D((3, 3), strides=(2, 2), name='pool5_1')(conv5_1)
    pool5_2 = MaxPooling2D((3, 3), strides=(2, 2), name='pool5_2')(conv5_2)

    # merge合并层：第五层进入全连接之前，要将分开的合并
    merge2 = concatenate([pool5_1, pool5_2], axis=1)

    # 通过flatten将多维输入一维化
    dense1 = Flatten(name='flatten')(merge2)

    # layer6, layer7 第六，七层， 进行两次4096维的全连接，中间加dropout避免过拟合
    dense2_1 = Dense(4096, activation='relu', name='dense2_1')(dense1)
    dense2_2 = Dropout(0.5)(dense2_1)

    dense3_1 = Dense(4096, activation='relu', name='dense3_1')(dense2_2)
    dense3_2 = Dropout(0.5)(dense3_1)

    # 输出层：输出类别，分类函数使用softmax
    dense3_3 = Dense(384, name='dense3_3')(dense3_2)

    # dense3_3 = Dense(nb_classes, name='dense3_3')(dense3_2)
    # prediction = Activation('softmax', name='softmax')(dense3_3)
    #
    # # 最后定义模型输出
    AlexNet = Model(input=inputs, outputs=dense3_3)
    # if (w_path):
    #     # 加载权重数据
    #     AlexNet.load_weights(w_path)
    AlexNet.summary()
    return AlexNet


from keras.layers import *
from keras.models import Model
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from config import *

def get_model(config_path, checkpoint_path, train_flag=1):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    for l in bert_model.layers:
        l.trainable = True

    T1 = Input(shape=(None,))
    T2 = Input(shape=(None,))
    input_shape = (128, 128, 3)
    T3 = Input(shape=input_shape, name='input')
    # 第一层：两个卷积操作和两个pooling操作

    conv1_1 = Convolution2D(48, (11, 11), strides=(4, 4), activation='relu', name='conv1_1')(T3)
    conv1_2 = Convolution2D(48, (11, 11), strides=(4, 4), activation='relu', name='conv1_2')(T3)

    pool1_1 = MaxPooling2D((3, 3), strides=(2, 2), name='pool1_1')(conv1_1)
    pool1_2 = MaxPooling2D((3, 3), strides=(2, 2), name='pool1_2')(conv1_2)

    # layer2:两个卷积，将前边的池化得到的数据，进行卷积，再继续池化

    conv2_1 = Convolution2D(128, (5, 5), activation='relu', padding='same')(pool1_1)
    conv2_2 = Convolution2D(128, (5, 5), activation='relu', padding='same')(pool1_2)

    pool2_1 = MaxPooling2D((3, 3), strides=(2, 2), name='pool2_1')(conv2_1)
    pool2_2 = MaxPooling2D((3, 3), strides=(2, 2), name='pool2_2')(conv2_2)

    # merge合并层：第二层进入第三层，将数据混合合并
    merge1 = concatenate([pool2_2, pool2_1], axis=1)

    # layer3:两个卷积操作

    conv3_1 = Convolution2D(192, (3, 3), activation='relu', name='conv3_1', padding='same')(merge1)
    conv3_2 = Convolution2D(193, (3, 3), activation='relu', name='conv3_2', padding='same')(merge1)

    # latyer4:两个卷积操作
    conv4_1 = Convolution2D(192, (3, 3), activation='relu', name='conv4_1', padding='same')(conv3_1)
    conv4_2 = Convolution2D(192, (3, 3), activation='relu', name='conv4_2', padding='same')(conv3_2)

    # layer5:两个卷积操作和两个pooling操作
    conv5_1 = Convolution2D(128, (3, 3), activation='relu', name='conv5_1', padding='same')(conv4_1)
    conv5_2 = Convolution2D(128, (3, 3), activation='relu', name='conv5_2', padding='same')(conv4_2)

    pool5_1 = MaxPooling2D((3, 3), strides=(2, 2), name='pool5_1')(conv5_1)
    pool5_2 = MaxPooling2D((3, 3), strides=(2, 2), name='pool5_2')(conv5_2)

    # merge合并层：第五层进入全连接之前，要将分开的合并
    merge2 = concatenate([pool5_1, pool5_2], axis=1)

    # 通过flatten将多维输入一维化
    dense1 = Flatten(name='flatten')(merge2)

    # layer6, layer7 第六，七层， 进行两次4096维的全连接，中间加dropout避免过拟合
    dense2_1 = Dense(4096, activation='relu', name='dense2_1')(dense1)
    dense2_2 = Dropout(0.5)(dense2_1)

    dense3_1 = Dense(4096, activation='relu', name='dense3_1')(dense2_2)
    dense3_2 = Dropout(0.5)(dense3_1)

    # 输出层：输出类别，分类函数使用softmax
    dense3_3 = Dense(384, name='dense3_3')(dense3_2)

    T = bert_model([T1, T2])
    T = Lambda(lambda x: x[:, 0])(T)

    seed = 2020
    T = concatenate([T, dense3_3])
    T = Dense(384, activation='relu')(T)
    output = Dense(nclass, activation='softmax')(T)

    model = Model([T1, T2, T3], output)
    if train_flag == 1:
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(2e-5),  # 用足够小的学习率
            metrics=['accuracy']
        )
    else:
        model = multi_gpu_model(model, gpus= 2)  # 使用几张显卡n等于几
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(2e-5),  # 用足够小的学习率
            metrics=['accuracy']
        )
    return model


