import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from math import floor

np.random.seed(7)

from setup import config

feature_ind_length = config.parameter.get('feature.ind.length')
feature_ind_embedding_offset = config.parameter.get('feature.ind.embedding.offset')
feature_ind_embedding_limit = config.parameter.get('feature.ind.embedding.limit')

feature_ind_model_lstm_with_embedding_base_path = config.runtime.path('feature_ind_model_lstm_with_embedding_')


from src.pipeline.reader import reader_csv
data_path = config.data.path('train_ind_uniquified.csv')
_, x, y = reader_csv(data_path, frac=0.01)

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
ratio_pos_neg = 1/3

# 计算需要的估计器
n_estimator = round(ratio_pos_neg * size_neg / size_pos) + 1
print("估计器数目: ", n_estimator)
config.parameter.put('feature.ind.model.lstm-with-embedding.estimator.numbers', n_estimator)

# 构造模型, 逐个训练

seg_neg_begin = list(range(0, size_neg, int(round(size_neg/n_estimator))))
seg_neg_end = list(seg_neg_begin)
if len(seg_neg_begin) != 0:
    seg_neg_end.pop(0)
    seg_neg_end.append(size_neg)
seg_neg = list(zip(seg_neg_begin, seg_neg_end))


data = []
for b, e in seg_neg:
    data.append(
        (
            np.concatenate((x_pos, x_neg[b:e]), axis=0),
            np.concatenate((y_pos, y_neg[b:e]), axis=0)
        )
    )

nums_word = feature_ind_embedding_limit - feature_ind_embedding_offset
embedding_vecor_length = 32
for i, (_x, _y) in enumerate(data):
    permutation = np.random.permutation(_y.shape[0])
    _x = _x[permutation]
    _y = _y[permutation]
    print(_x)
    print(_y)

    config.parameter.put('feature.ind.model.lstm-with-embedding.runtime.'+str(i),
                         feature_ind_model_lstm_with_embedding_base_path + str(i))
    model = Sequential()
    model.add(Embedding(nums_word, embedding_vecor_length, input_length=feature_ind_length))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    model.fit(_x, _y, epochs=10, batch_size=8, verbose=1, class_weight={0: 1/3, 1: 1-1/3})

    model.save(feature_ind_model_lstm_with_embedding_base_path + str(i))

