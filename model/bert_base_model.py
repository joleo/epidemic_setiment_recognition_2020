# -*- coding: utf-8 -*-
"""
@Time ： 2020/3/15 13:57
@Auth ： joleo
@File ：bert_base_model.py
"""

from keras.layers import *
from keras.models import Model
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras_layer.layers_keras import MaskedConv1D,MaskedGlobalMaxPool1D,MaskedGlobalAveragePooling1D
from keras_layer.bert_embedding import KerasBertEmbedding
from keras import backend as K
from loss.mul_focal_loss import focal_loss

def mycrossentropy(y_true, y_pred, e=0.1):
    loss1 = K.categorical_crossentropy(y_true, y_pred)
    loss2 = K.categorical_crossentropy(K.ones_like(y_pred) / 3, y_pred)
    return (1-e)*loss1 + e*loss2
    
def get_last_4_emb_model(config_path, checkpoint_path, dict_path, layer_num, layer_indexes, train_flag=1):
    bert_input, bert_out = KerasBertEmbedding(config_path, checkpoint_path,dict_path,layer_num,layer_indexes).bert_encode()
    hidden_1 = Lambda(lambda x: x[:, 0])(bert_out)
    T = Dense(64, activation='relu')(hidden_1)
    
    output = Dense(3, activation='softmax')(T)

    model = Model(bert_input, output)
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
    #model.summary()
    return model

def get_last_5_emb_model(config_path, checkpoint_path, dict_path, layer_num, layer_indexes, train_flag=1):
    bert_input, bert_out = KerasBertEmbedding(config_path, checkpoint_path,dict_path,layer_num,layer_indexes).bert_encode()
    hidden_1 = Lambda(lambda x: x[:, 0])(bert_out)
    T = Dense(64, activation='relu')(hidden_1)
    
    output = Dense(3, activation='softmax')(T)

    model = Model(bert_input, output)
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
   # model.summary()
    return model

def get_last_2_emb_model(config_path, checkpoint_path, dict_path, layer_num, layer_indexes, train_flag=1):
    bert_input, bert_out = KerasBertEmbedding(config_path, checkpoint_path,dict_path,layer_num,layer_indexes).bert_encode()
    hidden_1 = Lambda(lambda x: x[:, 0])(bert_out)
    T = Dense(64, activation='relu')(hidden_1)
    
    output = Dense(3, activation='softmax')(T)

    model = Model(bert_input, output)
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

def get_all_emb_model(config_path, checkpoint_path, dict_path, layer_num, layer_indexes, train_flag=1):
    bert_input, bert_out = KerasBertEmbedding(config_path, checkpoint_path,dict_path,layer_num,layer_indexes).bert_encode()
    hidden_1 = Lambda(lambda x: x[:, 0])(bert_out)
    T = Dense(64, activation='relu')(hidden_1)
    
    output = Dense(3, activation='softmax')(T)

    model = Model(bert_input, output)
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


def get_last_4_emb_with_avfp_model(config_path, checkpoint_path, dict_path, layer_num, layer_indexes, train_flag=1):
    bert_input, bert_out = KerasBertEmbedding(config_path, checkpoint_path,dict_path,layer_num,layer_indexes).bert_encode()

    T_1 = Lambda(lambda x: x[:, 0])(bert_out)
    T_2 = Lambda(lambda x: x[:, -1])(bert_out)
    avg_pool = MaskedGlobalAveragePooling1D()(bert_out)

    #T_64 = Dense(64, activation='tanh')(T_1)
    T = Concatenate()([T_1,avg_pool,T_2])
    
    T = Dense(64, activation='relu')(T)
    output = Dense(3, activation='softmax')(T)

    model = Model(bert_input, output)
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

def get_bert_gru_model(config_path, checkpoint_path, dict_path, layer_num,layer_indexes, train_flag=1):
    from keras import backend as K
    bert_input, bert_out = KerasBertEmbedding(config_path, checkpoint_path,dict_path,layer_num,layer_indexes).bert_encode()
    hidden_1 = Lambda(lambda x: x[:, 0])(bert_out)
    hidden_2 = Lambda(lambda x: x[:, -1])(bert_out)
    encode = Bidirectional(GRU(units=128, return_sequences=True))(bert_out)
    encode = Bidirectional(GRU(units=128, return_sequences=True))(bert_out)
   # encode_1 = Reshape(-1, 64*2,1)(encode)
    #encode_1 = K.reshape(encode,(-1,64*2) ) 
    avg_pool = MaskedGlobalAveragePooling1D()(encode)
    max_pool = MaskedGlobalMaxPool1D()(encode)
    conc = concatenate([hidden_1,avg_pool, max_pool,hidden_2])
    #x = Dropout(0.5)(conc)
    x = Dense(64, activation='relu')(conc)
    output_layers = Dense(3, activation='softmax')(x)
    model = Model(bert_input, output_layers)
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

def get_rcnn_model(config_path, checkpoint_path, train_flag=1):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    for l in bert_model.layers:
        l.trainable = True

    T1 = Input(shape=(None,))
    T2 = Input(shape=(None,))

    T = bert_model([T1, T2])

    T_ = Bidirectional(LSTM(units=32, return_sequences=True))(T)
    T_ = Bidirectional(LSTM(units=32, return_sequences=True))(T_)
    t_embed_layer = MaskedConv1D(filters=64, kernel_size=3, padding='same', activation='relu')(T_)
    pool = MaskedGlobalMaxPool1D()(t_embed_layer)
    ave = MaskedGlobalAveragePooling1D()(t_embed_layer)
    T_2 = Add()([pool, ave])
    
    #T = Concatenate()([T, T3_])
#    T_2 = Dense(64, activation='relu')(T_2)

    output = Dense(3, activation='softmax')(T_2)

    model = Model([T1, T2], output)
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


def get_model(config_path, checkpoint_path, train_flag=1):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    for l in bert_model.layers:
        l.trainable = True

    T1 = Input(shape=(None,))
    T2 = Input(shape=(None,))

    T = bert_model([T1, T2])
    T = Lambda(lambda x: x[:, 0])(T)

    output = Dense(3, activation='softmax')(T)

    model = Model([T1, T2], output)
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

  