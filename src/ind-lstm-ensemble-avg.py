import numpy as np
import pandas as pd
from setup import config
from src.pipeline.reader import reader_csv


data_lstm_resullt_path = config.data.path('predict-ind-lstm-with-embedding')
data_lstm_ensemble_avg_path = config.data.path('predict-ind-lstm-with-embedding.ensemble.avg.result.csv')

ID, x, _ = reader_csv(data_lstm_resullt_path)

_y = np.mean(x, axis=1)

# test_data_path = config.data.path('test_ind_uniquified.csv')
# ID, _, _ = reader_csv(test_data_path)

data_lstm_ensemble_pd = pd.DataFrame({'id': ID, 'target': _y})

data_lstm_ensemble_pd.to_csv(data_lstm_ensemble_avg_path, index=False)


