import pandas as pd
import numpy as np
from setup import config
import math
import preprocesss.utils as pre_process_utils

from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from src.model.lstm import EmbeddedLSTM

np.random.seed(7)

reset = True

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
    config = config.cast('feature.'+feature)

    model_name = 'model.lstm-weighted'
    model_path = config.runtime.path(model_name)
    model_path_tag = model_name + '.save'
    model_checkpoint_path = config.runtime.path(
        model_name+'.checkpoint'
                   '.epoch-{epoch:02d}'
                   '.val_loss-{val_loss:.6f}'
                   '.val_y_acc-{val_y_acc:.6f}'
                   '.val_y_aux_acc-{val_y_aux_acc:.6f}')
    model_checkpoint_tag = model_name + '.checkpoint.save'

    model_runtime_data_path = config.runtime.path(model_name + '.data')
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
    # 读取数据
    data_clip = pd.read_csv(data_clip_path, low_memory=True)

    feature_column_type = pre_process_utils.get_column_type_pair_list(df=data_clip, prefix='ps_')

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
    column_name_list_feature_type_float = sorted(column_name_list_feature_type_float, reverse=False)

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
    data_train = data[data['set'] == 'train']

    config.parameter.put(model_name + ".columns.int", column_name_list_feature_type_int)
    config.parameter.put(model_name + ".columns.float", column_name_list_feature_type_float)

    ####################################################################################################################
    # 数据构造

    x_int = data_train[column_name_list_feature_type_int].values
    x_float = data_train[column_name_list_feature_type_float].values
    y = data_train['target'].values
    ID = data_train['id'].values

    idx_pos = y == 1
    y_pos = y[idx_pos]
    size_pos = len(y_pos)
    print("正例数目: ", size_pos)

    idx_neg = y == 0
    y_neg = y[idx_neg]
    size_neg = len(y_neg)
    print("反例数目: ", size_neg)

    ####################################################################################################################
    # 模型构造

    embedded_lstm_config = EmbeddedLSTM.Config()
    embedded_lstm_config.x_int_dim = len(column_name_list_feature_type_int)
    embedded_lstm_config.x_float_dim = len(column_name_list_feature_type_float)
    embedded_lstm_config.dropout = 0.1
    embedded_lstm_config.embedding_word_number = embedding_word_number
    embedded_lstm_config.embeding_vector_length = 80
    embedded_lstm_config.lstm_units = 400
    embedded_lstm_config.dense = [400, 400, 200]

    sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)

    def step_decay(epoch):
        initial_lrate = 0.1
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate

    lr_rate = LearningRateScheduler(step_decay)

    rate_pos = size_pos/(size_pos+size_neg)
    class_weight = {
        1: 1 - rate_pos,
        0: rate_pos
    }
    print("样本权重: ", class_weight)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, mode='min', patience=3, verbose=1)

    checkpoint = ModelCheckpoint(model_checkpoint_path, monitor='val_y_acc', save_best_only=False, mode='max', verbose=1)

    ####################################################################################################################
    # Run !

    if flag_running:
        print('断点继续, step={0}'.format(breakpoint))

    breakpoint = 0
    config.parameter.put(model_runtime_flag_running_tag, True)

    config.parameter.put(model_path_tag, model_path)

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

    lstm.model.fit(
        x={'x_int': x_int, 'x_float': x_float},
        y={'y': y, 'y_aux': y},
        class_weight=class_weight,
        epochs=20, batch_size=64, shuffle=True, validation_split=0.2,
        callbacks=[early_stopping, checkpoint],
        verbose=1,
    )

    lstm.model.save(model_path)

    config.parameter.put(model_runtime_flag_running_tag, False)
