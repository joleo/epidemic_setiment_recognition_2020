# -*- coding: utf-8 -*-
"""
@Time ： 2020/3/26 15:59
@Auth ： joleo
@File ：bert_with_nn.py
"""
from keras.layers import *
from keras.models import Model
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras_layer.layers_keras import MaskedConv1D,MaskedGlobalMaxPool1D,MaskedGlobalAveragePooling1D

def get_model_textcnn(config_path, checkpoint_path, nb_words, EMBEDDING_DIM,embedding_matrix, train_flag=1):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    for l in bert_model.layers:
        l.trainable = True

    T1 = Input(shape=(None,))
    T2 = Input(shape=(None,))
    T3 = Input(shape=(None,))

    embed_layer = Embedding(input_dim=nb_words, output_dim=EMBEDDING_DIM, weights=[embedding_matrix], mask_zero=True)(T3)
    T1_ = bert_model([T1, T2])
    T1_ = Lambda(lambda x: x[:, 0])(T1_)

    #embed_layer = Concatenate()([T1_, embed_layer])
    T = MaskedConv1D(filters=256, kernel_size=3, padding='same', activation='relu')(embed_layer)
    t_pool = MaskedGlobalMaxPool1D()(T)
    t_ave = MaskedGlobalAveragePooling1D()(T)
    tt = Add()([t_pool, t_ave])
    T_ = Concatenate()([T1_, tt])
   # T_ = Dropout(0.1)(T_)
    T_ = Dense(64, activation='relu')(T_)
    output = Dense(3, activation='softmax')(T_)

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


def get_model_rnn_cnn(config_path, checkpoint_path, nb_words, EMBEDDING_DIM, embedding_matrix, train_flag=1):
    import keras.backend as K

    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    for l in bert_model.layers:
        l.trainable = True

    T1 = Input(shape=(None,))
    T2 = Input(shape=(None,))
    T3 = Input(shape=(None,))

    # embedding1 = Embedding(len(vocabulary) + 2, 200, weights=[embedding_index], mask_zero=True)
    embed_layer = Embedding(input_dim=nb_words, output_dim=EMBEDDING_DIM, weights=[embedding_matrix], mask_zero=True)(T3)
    T = bert_model([T1, T2])

    T = Lambda(lambda x: x[:, 0])(T)
    #embed_layer = Concatenate(axis=-1)([embed_layer, T3_])
    embed_layer = Bidirectional(LSTM(units=32, return_sequences=True))(embed_layer)
    embed_layer = Bidirectional(LSTM(units=32, return_sequences=True))(embed_layer)
    T_ = MaskedConv1D(filters=64, kernel_size=3, padding='same', activation='relu')(embed_layer)
    pool = MaskedGlobalMaxPool1D()(T_)
    ave = MaskedGlobalAveragePooling1D()(T_)
    T_1 = Add()([pool, ave])
    T = Concatenate(axis=-1)([T_1, T])

   # T = Dropout(0.1)(T)
    T = Dense(32, activation='relu')(T)
    output = Dense(3, activation='softmax')(T)

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
    model.summary()
    return model

