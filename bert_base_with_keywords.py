#! -*- coding:utf-8 -*-
import json
import numpy as np
from tqdm import tqdm
import time
import logging
from sklearn.model_selection import StratifiedKFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os
import pandas as pd
import re
import jieba
from keras.utils.np_utils import to_categorical
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score


learning_rate = 5e-5
min_learning_rate = 2e-5
flag = 1
MAX_LEN = 180
n_fold = 6
epoch = 1
weight_path = '/data/data01/liyang099/com/weight/chinese/'
if flag == 1:
    # base bert
    config_path = weight_path+ '/chinese_wwm_ex_L12/bert_config.json'
    checkpoint_path = weight_path+ '/chinese_wwm_ex_L12/bert_model.ckpt'
    dict_path =  weight_path+ '/chinese_wwm_ex_L12/vocab.txt'
elif flag == 2:
    config_path = weight_path+ '/chinese_roberta/bert_config.json'
    checkpoint_path = weight_path+ '/chinese_roberta/bert_model.ckpt'
    dict_path =  weight_path+ '/chinese_roberta/vocab.txt'
    dict_path =  weight_path+ '/chinese_wwm_ex_L12/vocab.txt'
elif flag == 3:
    config_path = weight_path+ '/chinese_wwm_ex_bert/bert_config.json'
    checkpoint_path = weight_path+ '/chinese_wwm_ex_bert/bert_model.ckpt'
    dict_path =  weight_path+ '/chinese_wwm_ex_bert/vocab.txt'
    dict_path =  weight_path+ '/chinese_wwm_ex_bert/vocab.txt'
elif flag == 4:
    config_path = weight_path+ '/chinese_roberta_wwm_large/bert_config.json'
    checkpoint_path = weight_path+ '/chinese_roberta_wwm_large/bert_model.ckpt'
    dict_path =  weight_path+ '/chinese_roberta_wwm_large/vocab.txt'

# token_dict = {}
# with open(dict_path, 'r', encoding='utf-8') as reader:
#     for line in reader:
#         token = line.strip()
#         token_dict[token] = len(token_dict)
# tokenizer = Tokenizer(token_dict)

########################################模型部分#############################################
import codecs

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)  # 给每个token 按序编号


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict)
##################################################################
file_path = '/data/data01/liyang099/com/setiment_regnition/data/logs/'
# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

# 创建一个handler，
timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
fh = logging.FileHandler(file_path + 'log_' + timestamp +'.txt')
fh.setLevel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)
############################################################################################
train= pd.read_pickle('data/train.pkl')
test= pd.read_pickle('data/test.pkl')


train['content'] = list(map(lambda x,y: str(x) + '' + str(y), train['key_words'], train['微博中文内容'])) 
test['content'] = list(map(lambda x,y: str(x) + ' ' + str(y), test['key_words'], test['微博中文内容'])) 
train['content'] = train['content'].map(lambda x: x.replace('\'',''))
test['content'] = test['content'].map(lambda x: x.replace('\'',''))

MAX_LEN = 180
train_achievements = train['content'].astype('str').values

labels = train['情感倾向'].astype(int).values+ 1
labels_cat = to_categorical(labels)
labels_cat = labels_cat.astype(np.int32)

test_achievements = test['content'].astype('str').values
############################################################################################
'填充序列长度'
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])



class data_generator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data[0]) // self.batch_size
        if len(self.data[0]) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            X1, y = self.data
            idxs = list(range(len(self.data[0])))
            np.random.shuffle(idxs)
            T, T_, Y = [], [], []
            for c, i in enumerate(idxs):
                achievements = X1[i]
#                 requirements = X2[i]
                t, t_ = tokenizer.encode(first=achievements, max_len=MAX_LEN)
                T.append(t)
                T_.append(t_)
                Y.append(y[i])
                if len(T) == self.batch_size or i == idxs[-1]:
                    T = seq_padding(T)
                    T_ = seq_padding(T_)
                    Y = np.array(Y)
                    yield [T, T_], Y
                    T, T_, Y = [], [], []


#############################################################################################
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback


def get_model():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    for l in bert_model.layers:
        l.trainable = True

    T1 = Input(shape=(None,))
    T2 = Input(shape=(None,))

    T = bert_model([T1, T2])

    T = Lambda(lambda x: x[:, 0])(T)

    output = Dense(3, activation='softmax')(T)

    model = Model([T1, T2], output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        metrics=['accuracy']
    )
    model.summary()
    return model


class Evaluate(Callback):
    def __init__(self, val_data, val_index):
        self.score = []
        self.best = 0.
        self.early_stopping = 0
        self.val_data = val_data
        self.val_index = val_index
        self.predict = []
        self.lr = 0
        self.passed = 0

    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低
        """
        if self.passed < self.params['steps']:
            self.lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, self.lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            self.lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            self.lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, self.lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        score, acc, f1 = self.evaluate()
        if score > self.best:
            self.best = score
            self.early_stopping = 0
           # model.save_weights('./data/model_save/bert{}.w'.format(fold))
        else:
            self.early_stopping += 1
        logger.info('lr: %.6f, epoch: %d, score: %.4f, acc: %.4f, f1: %.4f,best: %.4f\n' % (self.lr, epoch, score, acc, f1, self.best))

    def evaluate(self):
        self.predict = []
        prob = []
        val_x1, val_y, val_cat = self.val_data
        for i in tqdm(range(len(val_x1))):
            achievements = val_x1[i]
#             requirements = val_x2[i]

            t1, t1_ = tokenizer.encode(first=achievements)
            T1, T1_ = np.array([t1]), np.array([t1_])
            _prob = model.predict([T1, T1_])
            oof_train[self.val_index[i]] = _prob[0]
            self.predict.append(np.argmax(_prob, axis=1)[0]+1)
            prob.append(_prob[0])

        score = f1_score(val_y+1, self.predict, average='macro')#1.0 / (1 + mean_absolute_error(val_y+1, self.predict))
        acc = accuracy_score(val_y+1, self.predict)
        f1 = f1_score(val_y+1, self.predict, average='macro')
        return score, acc, f1


def predict(data):
    prob = []
    val_x1 = data
    for i in tqdm(range(len(val_x1))):
        achievements = val_x1[i]
#         requirements = val_x2[i]

        t1, t1_ = tokenizer.encode(first=achievements)
        T1, T1_ = np.array([t1]), np.array([t1_])
        _prob = model.predict([T1, T1_])
        prob.append(_prob[0])
    return prob
#############################################################################################
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)

oof_train = np.zeros((len(train), 3), dtype=np.float32)
oof_test = np.zeros((len(test), 3), dtype=np.float32)
for fold, (train_index, valid_index) in enumerate(skf.split(train_achievements, labels)):
    logger.info('================     fold {}        ==============='.format(fold))
    x1 = train_achievements[train_index]
    y = labels_cat[train_index]

    val_x1 = train_achievements[valid_index]
    val_y = labels[valid_index]
    val_cat = labels_cat[valid_index]

    train_D = data_generator([x1, y])
    evaluator = Evaluate([val_x1, val_y, val_cat], valid_index)

    model = get_model()
    model.fit_generator(train_D.__iter__(),
                        steps_per_epoch=len(train_D),
                        epochs=epoch,
                        callbacks=[evaluator]
                       )
    model.load_weights('./data/model_save/bert{}.w'.format(fold))
    oof_test += predict(test_achievements)
    K.clear_session()
#############################################################################################

oof_test /= n_fold
train['y'] =  np.argmax(oof_train, axis=1)
cv_score =f1_score(labels, train['y'], average='macro')
print(cv_score)
np.savetxt('./data/model_save/train_bert_prob_{}.txt'.format(cv_score), oof_train)
np.savetxt('./data/model_save/test_bert_prob_{}.txt'.format(cv_score), oof_test)
test['y'] = np.argmax(oof_test, axis=1) - 1
test['id'] = test['微博id'].copy()#map(lambda x: str(x)+' ')
test[['id', 'y']].to_csv('./data/submit/bert_{}.csv'.format(cv_score), index=False)
