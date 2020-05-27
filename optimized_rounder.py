# -*- coding: utf-8 -*-
"""
@Time ： 2020/3/14 9:37
@Auth ： joleo
@File ：optimized_rounder.py
"""
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from functools import partial
import scipy as sp

class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize Quadratic Weighted Kappa (QWK) score
    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved
    # https://www.kaggle.com/teejmahal20/regression-with-optimized-rounder
    """

    def __init__(self):
        self.coef_ = 0

    def _macro_f1_score(y_true, y_pred, n_labels):
        # https://www.kaggle.com/corochann/fast-macro-f1-computation
        total_f1 = 0.
        for i in range(n_labels):
            yt = y_true == i
            yp = y_pred == i

            tp = np.sum(yt & yp)

            tpfp = np.sum(yp)
            tpfn = np.sum(yt)
            if tpfp == 0:
                print('[WARNING] F-score is ill-defined and being set to 0.0 in labels with no predicted samples.')
                precision = 0.
            else:
                precision = tp / tpfp
            if tpfn == 0:
                print(f'[ERROR] label not found in y_true...')
                recall = 0.
            else:
                recall = tp / tpfn
            if precision == 0. or recall == 0.:
                f1 = 0.
            else:
                f1 = 2 * precision * recall / (precision + recall)
            total_f1 += f1
        return total_f1 / n_labels

    def _f1_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients

        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[0, 1, 2])

        return -f1_score(y, X_p, average='macro')

    def fit(self, X, y):
        """
        Optimize rounding thresholds

        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._f1_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds

        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[0, 1, 2])

    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']


optR = OptimizedRounder()
optR.fit(oof.reshape(-1,), train_y)
coefficients = optR.coefficients()
print(coefficients)
f1_score(train_y, np.round(oof), average = 'macro')

opt_preds = optR.predict(oof.reshape(-1,), coefficients)
f1_score(train_y, opt_preds, average = 'macro')

n_fold = 6
prediction = []
prediction = prediction / n_fold
prediction[prediction <= coefficients[0]] = 0
prediction[np.where(np.logical_and(prediction > coefficients[0], prediction <= coefficients[1]))] = 1
prediction[np.where(np.logical_and(prediction > coefficients[1], prediction <= coefficients[2]))] = 2
prediction[np.where(np.logical_and(prediction > coefficients[2], prediction <= coefficients[3]))] = 3
prediction[prediction > coefficients[9]] = 10

sample_df = pd.read_csv("/sample_submission.csv", dtype={'time':str})

sample_df['open_channels'] = prediction.astype(np.int)
sample_df.to_csv("submission.csv", index=False, float_format='%.4f')





import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import numpy as np

train = pd.read_csv('./train.csv',sep='  ',engine='python',encoding='utf8')
test = pd.read_csv('./test.csv',sep='  ',engine='python',encoding='utf8')
print(train.shape)
train = train.dropna()

train = train.reset_index(drop=True)
print(train.shape)
label_unique = train['stance'].unique()
nb_class = len(label_unique)

label_map = {'FAVOR':2, 'AGAINST':1, 'NONE':0}
label_map_r = {2:'FAVOR', 1:'AGAINST', 0:'NONE'}

import jieba
train['text_cut'] = train['text'].apply(lambda x:' '.join(jieba.cut(x)))
test['text_cut'] = test['text'].apply(lambda x:' '.join(jieba.cut(x)))

train_text = list(train['text_cut'].values)
test_text = list(test['text_cut'].values)
totle_text = train_text + test_text

tf = TfidfVectorizer(min_df=0, ngram_range=(1,2),stop_words='english')
tf.fit(totle_text)
train['stance'] = train['stance'].map(label_map)
X = tf.transform(train_text)
y = train['stance'].values
X_test = tf.transform(test_text)
skf = StratifiedKFold(n_splits=5,random_state=42,shuffle=True)

oof_train = np.zeros((len(train),nb_class))
oof_test = np.zeros((len(test),nb_class))

for idx,(tr_in,te_in) in enumerate(skf.split(X,y)):
    X_train = X[tr_in]
    X_valid = X[te_in]
    y_train = y[tr_in]
    y_valid = y[te_in]
    #
    # lr = LogisticRegression()

    lr1 = MultinomialNB(alpha=.25)

    # lr = LinearSVC()

    lr1.fit(X_train,y_train)
    # lr.fit(X_train,y_train)

    y_pred = lr1.predict_proba(X_valid)
    oof_train[te_in] = y_pred

    oof_test = oof_test + lr1.predict_proba(X_test) / skf.n_splits

# 使用包大人推荐的方法
x1 = np.array(oof_train)
y1 = np.array(y)
from scipy import optimize
def fun(x):
    tmp = np.hstack([x[0] * x1[:, 0].reshape(-1, 1), x[1] * x1[:, 1].reshape(-1, 1), x[2] * x1[:, 2].reshape(-1, 1)])
    return - accuracy_score(y1, np.argmax(tmp, axis=1))
x0 = np.asarray((0,0,0))
res = optimize.fmin_powell(fun, x0)

xx_score = accuracy_score(y,np.argmax(oof_train,axis=1))
print('原始score',xx_score)
# bestWght = search_best(oof_train, y)
xx_cv = accuracy_score(y,np.argmax(oof_train * res,axis=1))
print('修正后的',xx_cv)

result = test[['text']].copy()
result['label'] = np.argmax(oof_test,axis=1)
result['label2'] = np.argmax(oof_test*res,axis=1)
# print(result)
# result['id'] = test.index + 1
result['label'] = result['label'].map(label_map_r)
result['label2'] = result['label2'].map(label_map_r)

result[['label']].to_csv('./lr_tfidf_{}.csv'.format(str(xx_score).split('.')[1]),header=None,)
result[['label2']].to_csv('./lr_tfidf_{}.csv'.format(str(xx_cv).split('.')[1]),header=None,)