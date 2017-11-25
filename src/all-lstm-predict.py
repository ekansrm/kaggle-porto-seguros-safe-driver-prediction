import pandas as pd
import numpy as np
from setup import config
import math
import preprocesss.utils as preprocess_utils

from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from src.model.lstm_v2 import EmbeddedLSTM
from src.metric.gini import gini_callback

np.random.seed(7)

reset = False

if __name__ == '__main__':

    ####################################################################################################################
    # 数据全局变量
    data_path = config.data.path('data_indexed_train.csv')
    embedding_index_offset = config.parameter.get('embedding.index.offset')
    embedding_index_length = config.parameter.get('embedding.index.length')
    embedding_word_number = embedding_index_length - embedding_index_offset

    ####################################################################################################################
    # 设置
    feature = 'all'
    config = config.cast('feature.' + feature)

    model_name = 'model.lstm'
    model_path = config.runtime.path(model_name)
    model_path_tag = model_name + '.%d.save'

    model_checkpoint_best_path = config.runtime.path(model_name + '.%d.checkpoint.best')
    model_checkpoint_best_path_tag = model_name + '.checkpoint.best.save'

    model_predict_data_path = config.data.path(model_name + '.predict.train.%d.csv')

    ####################################################################################################################

    column_name_list_feature_type_int = config.parameter.get(model_name + ".columns.int")
    column_name_list_feature_type_float = config.parameter.get(model_name + ".columns.float")

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
    data_test = data[data['set'] == 'train']

    x_int = data_test[column_name_list_feature_type_int].values
    x_float = data_test[column_name_list_feature_type_float].values
    y = data_test['target'].values
    ID = data_test['id'].values

    embedded_lstm_config = EmbeddedLSTM.Config()
    embedded_lstm_config.x_int_dim = len(column_name_list_feature_type_int)
    embedded_lstm_config.x_float_dim = len(column_name_list_feature_type_float)
    embedded_lstm_config.dropout = 0.1
    embedded_lstm_config.embedding_word_number = embedding_word_number
    embedded_lstm_config.embeding_vector_length = 120
    embedded_lstm_config.lstm_units = 800
    embedded_lstm_config.dense = [400, 200]

    ####################################################################################################################
    # Run !

    for i in range(0, 4):
        lstm = EmbeddedLSTM()
        lstm.config = embedded_lstm_config

        lstm.model = model_checkpoint_best_path % i
        print("加载模型成功")

        lstm.model.summary()

        y = lstm.model.predict(
            x={'x_int': x_int, 'x_float': x_float},
            batch_size=32,
            verbose=1
        )
        y = y.reshape([-1])

        result = pd.DataFrame({'id': ID, 'target': y})

        result.to_csv(model_predict_data_path % i, index=False, float_format='%.8f')
