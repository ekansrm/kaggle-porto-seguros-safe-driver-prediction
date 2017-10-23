import numpy as np
from keras import optimizers
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=2)

np.random.seed(7)

from setup import config

feature_ps_ind_length = config.parameter.get('feature.ps_ind.length')
feature_ps_ind_embedding_offset = config.parameter.get('feature.ps_ind.embedding.offset')
feature_ps_ind_embedding_limit = config.parameter.get('feature.ps_ind.embedding.limit')

feature_ps_ind_model_lstm_with_embedding_path = config.runtime.path('feature_ps_ind_model_lstm_with_embedding_layer_2')

config.parameter.put('feature.ps_ind.model.lstm-with-embedding.runtime', feature_ps_ind_model_lstm_with_embedding_path)

rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)

nums_word = feature_ps_ind_embedding_limit - feature_ps_ind_embedding_offset
embedding_vecor_length = 32

model = Sequential()
model.add(Embedding(nums_word, embedding_vecor_length, input_length=feature_ps_ind_length))
model.add(Dropout(0.2))
model.add(LSTM(100, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['acc', 'mae'])

print(model.summary())

model.save(feature_ps_ind_model_lstm_with_embedding_path)

from setup import config
from src.pipeline.reader import reader_csv
data_path = config.data.path('train_ps_ind_uniquified.csv')
_, x, y = reader_csv(data_path)
model.fit(x, y, epochs=1, batch_size=2, class_weight={0: 0.034, 1: 1-0.034})

model.save(feature_ps_ind_model_lstm_with_embedding_path)


