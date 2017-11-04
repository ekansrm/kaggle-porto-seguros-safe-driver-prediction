import pandas as pd
import numpy as np
from setup import config
import preprocesss.utils as pre_process_utils

from keras.models import load_model

np.random.seed(7)

reset = True

if __name__ == '__main__':

    ####################################################################################################################
    # 数据全局变量
    data_clip_path = config.data.path('data_indexed_train_clip.csv')
    data_path = config.data.path('data_indexed_test.csv')
    embedding_index_offset = config.parameter.get('embedding.index.offset')
    embedding_index_length = config.parameter.get('embedding.index.length')
    embedding_word_number = embedding_index_length - embedding_index_offset

    ####################################################################################################################
    # 设置
    feature = 'all'
    config = config.cast('feature.' + feature)

    model_name = 'model.lstm'
    model_path = config.runtime.path('model.lstm.1.checkpoint.best')
    model_path_tag = model_name + '.save'
    model_checkpoint_path = config.runtime.path(
        model_name + '.checkpoint'
                     '.epoch-{epoch:02d}'
                     '.val_loss-{val_loss:.6f}'
                     '.val_y_acc-{val_y_acc:.6f}'
                     '.val_y_aux_acc-{val_y_aux_acc:.6f}')
    model_checkpoint_tag = model_name + '.checkpoint.save'

    model_runtime_flag_running_tag = model_name + '.runtime.flag.running'
    model_runtime_breakpoint_tag = model_name + '.runtime.breakpoint'
    model_predict_data_path = config.data.path(model_name + '.predict.csv')

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
    data_test = data[data['set'] == 'test']

    config.parameter.put(model_name + ".columns.int", column_name_list_feature_type_int)
    config.parameter.put(model_name + ".columns.float", column_name_list_feature_type_float)

    ####################################################################################################################
    # 数据构造

    x_int = data_test[column_name_list_feature_type_int].values
    x_float = data_test[column_name_list_feature_type_float].values
    y = data_test['target'].values
    ID = data_test['id'].values

    ####################################################################################################################
    # 预测

    model = load_model(model_path)
    _y, _y_aux = model.predict({'x_int': x_int, 'x_float': x_float}, verbose=True)
    _y = _y.reshape([-1])
    _y_aux = _y_aux.reshape([-1])

    result = pd.DataFrame({'id': ID, 'target': _y})

    result.to_csv(model_predict_data_path, index=False, float_format='%.8f')

    result = pd.DataFrame({'id': ID, 'target': _y_aux})

    result.to_csv(model_predict_data_path + '.aux', index=False, float_format='%.8f')
