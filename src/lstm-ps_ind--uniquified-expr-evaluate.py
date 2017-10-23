
from keras.models import load_model

from setup import config
import pandas as pd

feature_ps_ind_model_lstm_with_embedding_path = config.parameter.get('feature.ps_ind.model.lstm-with-embedding.runtime')


from src.pipeline.reader import reader_csv
data_path = config.data.path('train_ps_ind_uniquified.csv')

ID, x, y = reader_csv(data_path)

model = load_model(feature_ps_ind_model_lstm_with_embedding_path)
scores = model.evaluate(x, y, verbose=True)
print("Accuracy: %.2f%%" % (scores[1] * 100))




