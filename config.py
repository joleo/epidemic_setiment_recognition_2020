# -*- coding: utf-8 -*-
"""
@Time ： 2020/3/15 13:56
@Auth ： joleo
@File ：config.py
"""
import tensorflow as tf
import os

# 模型参数
nclass = 3
learning_rate =5e-5
min_learning_rate = 1e-5
flag = 8
batch_size = 16
train_flag = 1
MAX_LEN = 200
n_fold = 6
epoch = 3
layer_num = 12
#layer_indexes=[] # 默认为空，直接输出最后CLS层
layer_indexes = [-1,-2,-3,-4]

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

file_path = '/data/data01/liyang099/com/multi_setiment_reg/data/logs/'
weight_path = '/data/data01/liyang099/com/weight/chinese/'

# 图片参数
IMG_CHANNELS=3 #PGB图像为三通道 R、G、B
#weight_decay = 0.0005
IMG_ROWS=128  #图像的行像素
IMG_COLS=128  #图像的列像素
BATCH_SIZE=64 #batch大小
NB_EPOCH=10   #循环次数
NB_CLASSES=3  #分类  猫和狗两种
VERBOSE=1
#VALIDATION_SPLIT=0.2
#OPTIM=RMSprop()

# 权重设置
if flag == 1:
    # base bert
    config_path = weight_path+ '/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = weight_path+ '/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
    dict_path =  weight_path+ '/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt'
elif flag == 2:
    config_path = weight_path+ '/chinese_roberta_wwm_ext_L-12/bert_config.json'
    checkpoint_path = weight_path+ '/chinese_roberta_wwm_ext_L-12/bert_model.ckpt'
    dict_path =  weight_path+ '/chinese_roberta_wwm_ext_L-12/vocab.txt'
elif flag == 3:
    config_path = weight_path+ '/chinese_wwm_ex_bert/bert_config.json'
    checkpoint_path = weight_path+ '/chinese_wwm_ex_bert/bert_model.ckpt'
    dict_path =  weight_path+ '/chinese_wwm_ex_bert/vocab.txt'
elif flag == 4:
    config_path = weight_path+ '/chinese_roberta_wwm_large/bert_config.json'
    checkpoint_path = weight_path+ '/chinese_roberta_wwm_large/bert_model.ckpt'
    dict_path =  weight_path+ '/chinese_roberta_wwm_large/vocab.txt'
elif flag == 5:
    config_path = weight_path+ '/chinese_rbt3_L-3_H-768_A-12/bert_config_rbt3.json'
    checkpoint_path = weight_path+ '/chinese_rbt3_L-3_H-768_A-12/bert_model.ckpt'
    dict_path =  weight_path+ '/chinese_rbt3_L-3_H-768_A-12/vocab.txt'
elif flag == 6:
    config_path = weight_path+ '/chinese_rbtl3_L-3_H-1024_A-16/bert_config_rbtl3.json'
    checkpoint_path = weight_path+ '/chinese_rbtl3_L-3_H-1024_A-16/bert_model.ckpt'
    dict_path =  weight_path+ '/chinese_rbtl3_L-3_H-1024_A-16/vocab.txt'
elif flag == 7:
    weight_path = '/data/data01/liyang099/com/multi_setiment_reg/data/pretrain_weight/chinese_wwm_ext_l2_fur/'
    config_path = weight_path + 'bert_config.json'
    checkpoint_path = weight_path + 'bert_model.ckpt'
    dict_path = weight_path + 'vocab.txt'
elif flag == 8:
    weight_path = '/data/data01/liyang099/com/multi_setiment_reg/data/pretrain_weight/chinese_wwm_ext_l2_fur2/'
    config_path = weight_path + 'bert_config.json'
    checkpoint_path = weight_path + 'bert_model.ckpt'
    dict_path = weight_path + 'vocab.txt'
# token_dict = {}
# with open(dict_path, 'r', encoding='utf-8') as reader:
#     for line in reader:
#         token = line.strip()
#         token_dict[token] = len(token_dict)
# tokenizer = Tokenizer(token_dict)