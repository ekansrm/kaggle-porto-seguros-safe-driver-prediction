
from keras.models import load_model

from setup import config

feature_ind_model_lstm_with_embedding_path = config.parameter.get('feature.ind.model.lstm-with-embedding.runtime.0')


from src.pipeline.reader import reader_csv
data_path = config.data.path('train_ind_uniquified.csv')

ID, x, y = reader_csv(data_path, frac=0.01)

model = load_model(feature_ind_model_lstm_with_embedding_path)
scores = model.evaluate(x, y, verbose=True)
print("Accuracy: %.2f%%" % (scores[1] * 100))




