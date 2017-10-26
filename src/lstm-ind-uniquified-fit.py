import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from math import floor

np.random.seed(7)

from setup import config

feature_ind_length = config.parameter.get('feature.ind.length')
feature_ind_embedding_offset = config.parameter.get('feature.ind.embedding.offset')
feature_ind_embedding_limit = config.parameter.get('feature.ind.embedding.limit')

feature_ind_model_lstm_with_embedding_base_path = config.runtime.path('feature_ind_model_lstm_with_embedding_')


from src.pipeline.reader import reader_csv
data_path = config.data.path('train_ind_uniquified.csv')
_, x, y = reader_csv(data_path)

print(y)
y_pd = pd.DataFrame({'target': y})
ratio = y_pd['target'].value_counts() / y_pd['target'].size
print(ratio)
print(y_pd.size)


## 构造训练数据

idx_pos = y == 1
x_pos = x[idx_pos]
y_pos = y[idx_pos]
size_pos = len(y_pos)
print("正例数目: ", size_pos)

idx_neg = y == 0
x_neg = x[idx_neg]
y_neg = y[idx_neg]
size_neg = len(y_neg)
print("反例数目: ", size_neg)

# 正反例数据比
ratio_pos_neg = 1

# 计算需要的估计器
n_estimator = size_neg // size_pos
print("估计器数目: ", n_estimator)
config.parameter.put('feature.ind.model.lstm-with-embedding.estimator.numbers', n_estimator)

# 构造模型, 逐个训练

seg_neg_begin = list(range(0, size_neg, size_pos))
seg_neg_end = list(seg_neg_begin)
if len(seg_neg_begin) != 0:
    seg_neg_end.pop(0)
    seg_neg_end.append(size_neg)
seg_neg = list(zip(seg_neg_begin, seg_neg_end))
seg_neg = seg_neg[0: n_estimator]


data = []
for b, e in seg_neg:
    _x = np.concatenate((x_pos, x_neg[b:e]), axis=0)
    _y = np.concatenate((y_pos, y_neg[b:e]), axis=0)
    permutation = np.random.permutation(_y.shape[0])
    _x = _x[permutation]
    _y = _y[permutation]
    data.append((_x, _y))

nums_word = feature_ind_embedding_limit - feature_ind_embedding_offset
embedding_vecor_length = 32
early_stopping = EarlyStopping(monitor='loss', patience=2)
for i, (_x, _y) in enumerate(data):
    print(_x.size)
    print(_y.size)

    config.parameter.put('feature.ind.model.lstm-with-embedding.runtime.'+str(i),
                         feature_ind_model_lstm_with_embedding_base_path + str(i))
    model = Sequential()
    model.add(Embedding(nums_word, embedding_vecor_length, input_length=feature_ind_length))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mae'])

    print(model.summary())

    model.fit(_x, _y, epochs=10, batch_size=64, verbose=1, callbacks=[early_stopping])

    model.save(feature_ind_model_lstm_with_embedding_base_path + str(i))

