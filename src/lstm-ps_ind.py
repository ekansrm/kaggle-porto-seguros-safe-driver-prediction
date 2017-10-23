import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from setup import config

np.random.seed(7)

feature_ps_ind_model_lstm_path = config.runtime.path('feature_ps_ind_model_lstm')
config.parameter.put('feature.ps_ind.model.lstm.runtime', feature_ps_ind_model_lstm_path)

model = Sequential()
model.add(LSTM(name='lstm-1', units=100, input_shape=[31, 1], return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, name='lstm-2'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

from src.pipeline.reader import reader_csv
data_path = config.data.path('train_ps_ind.csv')
_, x, y = reader_csv(data_path)

x = x.reshape([-1, 31, 1])

model.fit(x, y, epochs=2, batch_size=32, class_weight={0: 0.034, 1: 1-0.034})

model.save(feature_ps_ind_model_lstm_path)
