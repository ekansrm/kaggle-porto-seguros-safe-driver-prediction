import pandas as pd
import numpy as np
from setup import config
import math
import preprocesss.utils as preprocess_utils

import pickle
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
from src.model.dense import TotalDense

np.random.seed(7)

reset = True
正反例数据比 = 1

if __name__ == '__main__':

    feature = ''

    训练数据文件clip路径 = config.data.path('data_indexed_train_clip.csv')
    训练数据文件路径 = config.data.path('data_indexed_train_clip.csv')
    embedding_index_offset = config.parameter.get('embedding.index.offset')
    embedding_index_length = config.parameter.get('embedding.index.length')

    config = config.cast('feature.'+feature)
    模型文件基地址 = config.runtime.path('model-dense-')
    估计器训练数据文件基地址 = config.runtime.path('model.dense.data.')
    running_config_tag = 'model.dense.runtime.running'
    断点标签 = 'model.dense.runtime.breakpoint'

    embedding_word_number = embedding_index_length - embedding_index_offset

    if reset:
        running = False
    else:
        running = config.parameter.get(running_config_tag)
        if running is None:
            running = False

    if running:
        上次运行断点 = config.parameter.get(断点标签)
        if 上次运行断点 is None:
            上次运行断点 = -1
    else:
        上次运行断点 = -1

    if not running:
        data_clip = pd.read_csv(训练数据文件clip路径, low_memory=True)

        feature_column_type = preprocess_utils.get_column_type_pair_list(df=data_clip, prefix='ps_'+feature)

        feature_column_name_list = [x[0] for x in feature_column_type]

        column_name_list_feature_type_int = [
            x[0]
            for x in filter(
                lambda x: 'float' not in x[1],
                feature_column_type
            )
        ]
        column_name_list_feature_type_int = sorted(column_name_list_feature_type_int, reverse=True)

        column_name_list_feature_type_float = [
            x[0]
            for x in filter(
                lambda x: 'float' in x[1],
                feature_column_type
            )
        ]
        column_name_list_feature_type_float = sorted(column_name_list_feature_type_float, reverse=True)

        dtype_dict = {
            'id': 'uint32',
            'target': 'float32',
            'set': 'str'
        }
        dtype_dict.update(
            {
                k: 'float32' for k in column_name_list_feature_type_float
            }
        )
        dtype_dict.update(
            {
                k: 'uint8' for k in column_name_list_feature_type_int
            }
        )

        data = pd.read_csv(训练数据文件路径, dtype=dtype_dict, low_memory=True)
        data_train = data[data['set'] == 'train']

        config.parameter.put("model.dense.columns", feature_column_name_list)
        config.parameter.put("model.dense.columns.int", column_name_list_feature_type_int)
        config.parameter.put("model.dense.columns.float", column_name_list_feature_type_float)

        x = data_train[feature_column_name_list].values
        y = data_train['target'].values
        ID = data_train['id'].values

        ## 构造训练数据

        idx_pos = y == 1
        x_pos = x[idx_pos]
        y_pos = y[idx_pos]
        ID_pos = ID[idx_pos]
        size_pos = len(y_pos)
        print("正例数目: ", size_pos)

        idx_neg = y == 0
        x_neg = x[idx_neg]
        y_neg = y[idx_neg]
        ID_neg = ID[idx_neg]
        size_neg = len(y_neg)
        print("反例数目: ", size_neg)

        # 计算需要的估计器
        估计器数量 = size_neg // size_pos
        print("估计器数目: ", 估计器数量)
        config.parameter.put('model.dense.estimator.numbers', 估计器数量)

        # 构造模型, 逐个训练

        seg_neg_begin = list(range(0, size_neg, size_pos))
        seg_neg_end = list(seg_neg_begin)
        if len(seg_neg_begin) != 0:
            seg_neg_end.pop(0)
            seg_neg_end.append(size_neg)
        seg_neg = list(zip(seg_neg_begin, seg_neg_end))
        seg_neg = seg_neg[0: 估计器数量]

        for i, (b, e) in enumerate(seg_neg):
            _x = np.concatenate((x_pos, x_neg[b:e]), axis=0)
            _y = np.concatenate((y_pos, y_neg[b:e]), axis=0)
            _id = np.concatenate((ID_pos, ID_neg[b:e]), axis=0)
            permutation = np.random.permutation(_y.shape[0])
            _x = _x[permutation]
            _y = _y[permutation]
            _id = _id[permutation]
            with open(估计器训练数据文件基地址 + str(i), 'wb') as fp:
                pickle.dump((_x, _y), fp, 1)


    else:
        # 从文件恢复数据
        估计器数量 = config.parameter.get('model.dense.estimator.numbers')
        feature_column_name_list = config.parameter.get("model.dense.columns")
        column_name_list_feature_type_int = config.parameter.get("model.dense.columns.int")
        column_name_list_feature_type_float = config.parameter.get("model.dense.columns.float")

    # 如果是训练, 恢复配置好的数据, 否则, 重新加载数据, 并保存为文件

    if running:
        print('断点继续, step={0}'.format(上次运行断点))

    dense_config = TotalDense.Config()
    dense_config.x_dim = len(feature_column_name_list)
    dense_config.y_dim = 1
    dense_config.dense = [800, 100]
    dense_config.dropout = 0.1

    sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)

    def step_decay(epoch):
        initial_lrate = 0.1
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate
    lrate = LearningRateScheduler(step_decay)

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)


    config.parameter.put(running_config_tag, True)
    for 当前步骤 in range(0, 估计器数量):

        if 当前步骤 < 上次运行断点:
            continue

        print('='*120)
        print('训练估计器... {0}/{1}'.format(当前步骤, 估计器数量))
        print('加载训练数据...')
        with open(估计器训练数据文件基地址 + str(当前步骤), 'rb') as fp:
            (_x, _y) = pickle.load(fp)

        config.parameter.put(断点标签, 当前步骤)
        config.parameter.put('model.dense.runtime.save.'+str(当前步骤), 模型文件基地址 + str(当前步骤))

        dense = TotalDense()
        dense.config = dense_config
        dense.build()
        dense.model.summary()

        dense.model.compile(
            optimizer='adam',
            loss={'y': 'binary_crossentropy'},
            loss_weights={'y': 1.},
            metrics=['accuracy']
        )

        dense.model.fit(
            x={'x': _x},
            y={'y': _y},
            epochs=1000, batch_size=16, shuffle=True, validation_split=0.2, verbose=1, callbacks=[]
        )

        dense.model.save(模型文件基地址 + str(当前步骤))

        print('='*120)

    config.parameter.put(running_config_tag, False)
