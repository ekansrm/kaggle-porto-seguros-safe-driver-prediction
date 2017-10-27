import pandas as pd
import numpy as np
from setup import config
import preprocesss.utils as preprocess_utils

from utils.ConfigHelper import Config
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping

np.random.seed(7)

reset = False
正反例数据比 = 1

if __name__ == '__main__':
    训练数据文件路径 = config.data.path('train_preprocessing_save_1_clip.csv')
    模型文件基地址 = config.runtime.path('feature_ind_model_lstm_with_embedding_')

    embedding_index_offset = config.parameter.get('feature.all.embedding.index.offset')
    embedding_index_length = config.parameter.get('feature.all.embedding.index.length')
    embedding_word_number = embedding_index_length - embedding_index_offset
    embeding_vector_length = 32

    if reset:
        training = False
    else:
        training = config.parameter.get("training")
        if training is None:
            training = False

    if not training:
        上次运行断点 = config.parameter.get('breakpoint')
    else:
        上次运行断点 = -1

    if not training:
        data_train = pd.read_csv(训练数据文件路径)
        feature_column_type = preprocess_utils.get_column_type_pair_list(df=data_train, prefix='ps')

        column_name_list_feature_type_int = [
            x[0]
            for x in filter(
                lambda x: 'float' not in x[1],
                feature_column_type
            )
        ]

        column_name_list_feature_type_float = [
            x[0]
            for x in filter(
                lambda x: 'float' not in x[1],
                feature_column_type
            )
        ]

        x = data_train[column_name_list_feature_type_int].values
        y = data_train['target'].values

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

        # 计算需要的估计器
        估计器数量 = size_neg // size_pos
        print("估计器数目: ", 估计器数量)
        config.parameter.put('feature.all.model.lstm-with-embedding.estimator.numbers', 估计器数量)

        # 构造模型, 逐个训练

        seg_neg_begin = list(range(0, size_neg, size_pos))
        seg_neg_end = list(seg_neg_begin)
        if len(seg_neg_begin) != 0:
            seg_neg_end.pop(0)
            seg_neg_end.append(size_neg)
        seg_neg = list(zip(seg_neg_begin, seg_neg_end))
        seg_neg = seg_neg[0: 估计器数量]

        估计器训练数据列表 = []
        for b, e in seg_neg:
            _x = np.concatenate((x_pos, x_neg[b:e]), axis=0)
            _y = np.concatenate((y_pos, y_neg[b:e]), axis=0)
            permutation = np.random.permutation(_y.shape[0])
            _x = _x[permutation]
            _y = _y[permutation]
            估计器训练数据列表.append((_x, _y))

    else:
        # 从文件恢复数据
        估计器数量 = config.parameter.get('feature.all.model.lstm-with-embedding.estimator.numbers')
        估计器训练数据列表 = []

    # 如果是训练, 恢复配置好的数据, 否则, 重新加载数据, 并保存为文件

    early_stopping = EarlyStopping(monitor='loss', patience=2)
    for 当前步骤, (_x, _y) in enumerate(估计器训练数据列表):

        if 当前步骤 < 上次运行断点:
            continue
        config.parameter.put('', 当前步骤)
        config.parameter.put('feature.all.model.lstm-with-embedding.runtime.'+str(当前步骤), 模型文件基地址 + str(当前步骤))
        model = Sequential()
        model.add(Embedding(embedding_word_number, embeding_vector_length, input_length=len(column_name_list_feature_type_int)))
        model.add(Dropout(0.2))
        model.add(LSTM(100))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mae'])

        print(model.summary())

        model.fit(_x, _y, epochs=10, batch_size=64, verbose=1, callbacks=[early_stopping])

        model.save(模型文件基地址 + str(当前步骤))
