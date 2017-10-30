import pandas as pd
import numpy as np
import math
from setup import config
import preprocesss.utils as preprocess_utils

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, log_loss

np.random.seed(7)

reset = False
正反例数据比 = 1

if __name__ == '__main__':

    训练数据文件路径 = config.data.path('train_preprocessing_save_1_clip.csv')

    config = config.cast('feature.all')
    模型文件路径基 = config.runtime.path('model_lstm_with_embedding_adjust_class_weight_')
    估计器训练数据文件基地址 = config.runtime.path('model.lstm-with-embedding-adjust-classs-weight.runtime.data.')
    running_config_tag = 'model.lstm-with-embedding-adjust-classs-weight.runtime.running'
    断点标签 = 'model.lstm-with-embedding-adjust-classs-weight.runtime.breakpoint'

    embedding_index_offset = config.parameter.get('embedding.index.offset')
    embedding_index_length = config.parameter.get('embedding.index.length')
    embedding_word_number = embedding_index_length - embedding_index_offset
    embeding_vector_length = 32

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

    data_train = pd.read_csv(训练数据文件路径)
    feature_column_type = preprocess_utils.get_column_type_pair_list(df=data_train, prefix='ps')

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
            lambda x: 'float' not in x[1],
            feature_column_type
        )
    ]
    column_name_list_feature_type_float = sorted(column_name_list_feature_type_float, reverse=True)

    config.parameter.put("model.lstm-with-embedding.columns.int", column_name_list_feature_type_int)
    config.parameter.put("model.lstm-with-embedding.columns.float", column_name_list_feature_type_float)

    x = data_train[column_name_list_feature_type_int].values
    y = data_train['target'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

    early_stopping = EarlyStopping(monitor='loss', patience=2)

    sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
    model = Sequential()
    model.add(Embedding(
        embedding_word_number, embeding_vector_length,
        input_length=len(column_name_list_feature_type_int)))
    model.add(Dropout(0.2))
    model.add(LSTM(512))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy', 'mae'])

    print(model.summary())

    kf = KFold(n_splits=10, shuffle=True, random_state=np.random.RandomState(31337))


    def step_decay(epoch):
        initial_lrate = 0.1
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate

    lrate = LearningRateScheduler(step_decay)

    config.parameter.put(running_config_tag, True)
    for i, (train_index, test_index) in enumerate(kf.split(x_train, y_train)):
        model.fit(x_train[train_index], y_train[train_index],
                  epochs=4, batch_size=32, class_weight={0: 0.034, 1: 1-0.034}, callbacks=[lrate])
        print(model.evaluate(x_train[test_index], y_train[test_index]))

        model.save(模型文件路径基 + str(i))

    config.parameter.put(running_config_tag, False)
