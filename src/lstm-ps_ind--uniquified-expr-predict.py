
from keras.models import load_model

from setup import config
import pandas as pd

feature_ps_ind_model_lstm_with_embedding_path = config.parameter.get('feature.ps_ind.model.lstm-with-embedding.runtime')


from src.pipeline.reader import reader_csv
data_path = config.data.path('test_ps_ind_uniquified.csv')

ID, x, _ = reader_csv(data_path)

model = load_model(feature_ps_ind_model_lstm_with_embedding_path)
_y = model.predict(x, verbose=True)
_y = _y.reshape([-1])


result = pd.DataFrame({'id': ID, 'target': _y})

path = config.data.path('predict-ps_ind-lstm-with-embedding-expr')

result.to_csv(path, index=False)


