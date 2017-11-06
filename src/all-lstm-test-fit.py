import pandas as pd
import numpy as np
from setup import config
import math
import preprocesss.utils as preprocess_utils

import pickle
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from src.model.lstm import EmbeddedLSTM
from src.metric.gini import gini_callback

np.random.seed(7)

reset = False

if __name__ == '__main__':

    ####################################################################################################################
    # 数据全局变量
    data_clip_path = config.data.path('data_indexed_train_clip.csv')
    data_path = config.data.path('data_indexed_train_clip.csv')
    embedding_index_offset = config.parameter.get('embedding.index.offset')
    embedding_index_length = config.parameter.get('embedding.index.length')
    embedding_word_number = embedding_index_length - embedding_index_offset

    ####################################################################################################################
    # 设置
    feature = 'all'
    config = config.cast('feature.' + feature)

    model_name = 'model.lstm.test'
    model_path = config.runtime.path(model_name + '.%d')
    model_path_tag = model_name + '.%d.save'
    model_checkpoint_path = config.runtime.path(
        model_name + '.%d.checkpoint'
                     '.epoch-{epoch:02d}'
                     '.val_loss-{val_loss:.6f}'
                     '.val_y_acc-{val_y_acc:.6f}'
                     '.val_y_aux_acc-{val_y_aux_acc:.6f}')

    model_checkpoint_best_path = config.runtime.path(model_name + '.%d.checkpoint.best')
    model_checkpoint_best_path_tag = model_name + '.%d.checkpoint.best.save'

    model_runtime_data_path = config.runtime.path(model_name + '.data.%d')
    model_runtime_flag_running_tag = model_name + '.runtime.flag.running'
    model_runtime_breakpoint_tag = model_name + '.runtime.breakpoint'

    if reset:
        flag_running = False
    else:
        flag_running = config.parameter.get(model_runtime_flag_running_tag)
        if flag_running is None:
            flag_running = False

    if flag_running:
        breakpoint = config.parameter.get(model_runtime_breakpoint_tag)
        if breakpoint is None:
            breakpoint = -1
    else:
        breakpoint = -1

    ####################################################################################################################

    if not flag_running:
        data_clip = pd.read_csv(data_clip_path, low_memory=True)

        feature_column_type = preprocess_utils.get_column_type_pair_list(df=data_clip, prefix='ps_')

        column_name_list_feature_type_int = [
            x[0]
            for x in filter(
                lambda x: 'float' not in x[1],
                feature_column_type
            )
        ]
        feature_ps_ind = sorted([x for x in column_name_list_feature_type_int if 'ps_ind' in x])
        feature_ps_calc = sorted([x for x in column_name_list_feature_type_int if 'ps_calc' in x])
        feature_ps_reg = sorted([x for x in column_name_list_feature_type_int if 'ps_reg' in x])
        feature_ps_car = sorted([x for x in column_name_list_feature_type_int if 'ps_car' in x])
        assert len(column_name_list_feature_type_int) == \
            len(feature_ps_ind + feature_ps_calc + feature_ps_reg + feature_ps_car)
        column_name_list_feature_type_int = feature_ps_ind + feature_ps_calc + feature_ps_reg + feature_ps_car

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

        data = pd.read_csv(data_path, dtype=dtype_dict, low_memory=True)
        data = data.sample(n=3000)
        data_train = data[data['set'] == 'train']

        config.parameter.put(model_name + ".columns.int", column_name_list_feature_type_int)
        config.parameter.put(model_name + ".columns.float", column_name_list_feature_type_float)

        x_int = data_train[column_name_list_feature_type_int].values
        x_float = data_train[column_name_list_feature_type_float].values
        y = data_train['target'].values
        ID = data_train['id'].values

        ## 构造训练数据

        idx_pos = y == 1
        x_int_pos = x_int[idx_pos]
        x_float_pos = x_float[idx_pos]
        y_pos = y[idx_pos]
        ID_pos = ID[idx_pos]
        size_pos = len(y_pos)
        print("正例数目: ", size_pos)

        idx_neg = y == 0
        x_int_neg = x_int[idx_neg]
        x_float_neg = x_float[idx_neg]
        y_neg = y[idx_neg]
        ID_neg = ID[idx_neg]
        size_neg = len(y_neg)
        print("反例数目: ", size_neg)

        # 计算需要的估计器
        n_estimator = size_neg // size_pos
        print("估计器数目: ", n_estimator)
        config.parameter.put(model_name + '.estimator.numbers', n_estimator)

        # 构造模型, 逐个训练

        seg_neg_begin = list(range(0, size_neg, size_pos))
        seg_neg_end = list(seg_neg_begin)
        if len(seg_neg_begin) != 0:
            seg_neg_end.pop(0)
            seg_neg_end.append(size_neg)
        seg_neg = list(zip(seg_neg_begin, seg_neg_end))
        seg_neg = seg_neg[0: n_estimator]

        for i, (b, e) in enumerate(seg_neg):
            _x_int = np.concatenate((x_int_pos, x_int_neg[b:e]), axis=0)
            _x_float = np.concatenate((x_float_pos, x_float_neg[b:e]), axis=0)
            _y = np.concatenate((y_pos, y_neg[b:e]), axis=0)
            _id = np.concatenate((ID_pos, ID_neg[b:e]), axis=0)
            permutation = np.random.permutation(_y.shape[0])
            _x_int = _x_int[permutation]
            _x_float = _x_float[permutation]
            _y = _y[permutation]
            _id = _id[permutation]
            with open(model_runtime_data_path % i, 'wb') as fp:
                pickle.dump((_x_int, _x_float, _y), fp, 1)


    else:
        # 从文件恢复数据
        n_estimator = config.parameter.get(model_name + '.estimator.numbers')
        column_name_list_feature_type_int = config.parameter.get(model_name + ".columns.int")
        column_name_list_feature_type_float = config.parameter.get(model_name + ".columns.float")

    # 如果是训练, 恢复配置好的数据, 否则, 重新加载数据, 并保存为文件

    ####################################################################################################################
    # 模型构造

    embedded_lstm_config = EmbeddedLSTM.Config()
    embedded_lstm_config.x_int_dim = len(column_name_list_feature_type_int)
    embedded_lstm_config.x_float_dim = len(column_name_list_feature_type_float)
    embedded_lstm_config.dropout = 0.1
    embedded_lstm_config.embedding_word_number = embedding_word_number
    embedded_lstm_config.embeding_vector_length = 80
    embedded_lstm_config.lstm_units = 800
    embedded_lstm_config.dense = [400, 400, 200]

    sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)

    def step_decay(epoch):
        initial_lrate = 0.1
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate
    lrate = LearningRateScheduler(step_decay)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, mode='min', patience=2, verbose=1)

    ####################################################################################################################
    # Run !
    if flag_running:
        print('断点继续, step={0}'.format(breakpoint))

    breakpoint = 0
    config.parameter.put(model_runtime_flag_running_tag, True)
    for i in range(0, n_estimator):

        if i < breakpoint:
            continue

        print('='*120)
        print('训练估计器... {0}/{1}'.format(i, n_estimator))
        print('加载训练数据...')
        with open(model_runtime_data_path % i, 'rb') as fp:
            (_x, _x_float, _y) = pickle.load(fp)

        config.parameter.put(model_runtime_breakpoint_tag, i)
        config.parameter.put(model_path_tag % i, model_path % i)

        lstm = EmbeddedLSTM()
        lstm.config = embedded_lstm_config
        lstm.build()
        lstm.model.summary()

        lstm.model.compile(
            optimizer='adam',
            loss={'y': 'binary_crossentropy', 'y_aux': 'binary_crossentropy'},
            loss_weights={'y': 1., 'y_aux': 0.7},
            metrics=['accuracy']
        )

        checkpoint = ModelCheckpoint(
            model_checkpoint_path % i, save_best_only=False, verbose=1)

        checkpoint_best = ModelCheckpoint(
            model_checkpoint_best_path % i, save_best_only=True, monitor='val_y_acc',  mode='max', verbose=1)

        lstm.model.fit(
            x={'x_int': _x, 'x_float': _x_float},
            y={'y': _y, 'y_aux': _y},
            epochs=20, batch_size=64, shuffle=True, validation_split=0.2, verbose=1,
            callbacks=[early_stopping, checkpoint, checkpoint_best, gini_callback()]
        )

        lstm.model.save(model_path % i)

        print('='*120)

    config.parameter.put(model_runtime_flag_running_tag, False)
