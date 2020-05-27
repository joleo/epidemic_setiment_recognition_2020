# -*- coding: utf-8 -*-
"""
@Time ： 2020/3/30 15:21
@Auth ： joleo
@File ：bert_embedding.py
"""
from keras_bert import load_trained_model_from_checkpoint
import keras.backend.tensorflow_backend as ktf_keras
import keras.backend as k_keras
from keras.models import Model
from keras.layers import Add,Dropout
import tensorflow as tf

# 全局使用，使其可以django、flask、tornado等调用
graph = None
model = None

class KerasBertEmbedding():
    def __init__(self,config_name,ckpt_name,dict_path,layer_num,layer_indexes=[]):
        self.config_path = config_name
        self.checkpoint_path = ckpt_name
        self.dict_path = dict_path
       # self.max_seq_len = max_seq_len
        self.layer_num = layer_num
        self.layer_indexes = layer_indexes

    def bert_encode(self):
        # 全局使用，使其可以django、flask、tornado等调用
        global graph
        graph = tf.get_default_graph()
        global model
        model = load_trained_model_from_checkpoint(self.config_path, self.checkpoint_path, seq_len=200)
        for l in model.layers:
            l.trainable = True
            
        print(model.output)
        print(len(model.layers))
        # lay = model.layers
        #一共104个layer，其中前八层包括token,pos,embed等，
        # 每8层（MultiHeadAttention,Dropout,Add,LayerNormalization）
        # 一共12层
        layer_dict = [7]
        layer_0 = 7
        for i in range(12):
            layer_0 = layer_0 + 8
            layer_dict.append(layer_0)
        # 输出它本身
        if len(self.layer_indexes) == 0:
            encoder_layer = model.output
        elif len(self.layer_indexes) == 1: # 分类如果只有一层，就只取最后那一层的weight；取得不正确，就默认取最后一层
            if self.layer_indexes[0] in [i+1 for i in range(self.layer_num)]:
                encoder_layer = model.get_layer(index=layer_dict[self.layer_indexes[0]-1]).output
            else:
                encoder_layer = model.get_layer(index=layer_dict[-1]).output
        else:  # 否则遍历需要取的层，把所有层的weight取出来并拼接起来shape:768*层数
            # layer_indexes must be [1,2,3,......12]
            # all_layers = [model.get_layer(index=lay).output if lay is not 1 else model.get_layer(index=lay).output[0] for lay in layer_indexes]
            all_layers = [model.get_layer(index=layer_dict[lay-1]).output if lay in [i+1 for i in range(self.layer_num)]
                          else model.get_layer(index=layer_dict[-1]).output  #如果给出不正确，就默认输出最后一层
                          for lay in self.layer_indexes]
            print(self.layer_indexes)
            #print(all_layers)
            all_layers_select = []
            for all_layers_one in all_layers:
                #all_layers_one = Dropout(0.5)(all_layers_one)
                all_layers_select.append(all_layers_one)
            encoder_layer = Add()(all_layers_select)
            print(encoder_layer.shape)
        print("KerasBertEmbedding:")
        print(encoder_layer.shape)
        #output_layer = NonMaskingLayer()(encoder_layer)
        model = Model(model.inputs, encoder_layer)
        # model.summary(120)
        return model.inputs, model.output
