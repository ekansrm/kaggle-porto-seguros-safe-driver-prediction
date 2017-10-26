import pandas as pd
from keras.models import load_model
from setup import config
from src.pipeline.reader import reader_csv

n_estimator = config.parameter.get('feature.ind.model.lstm-with-embedding.estimator.numbers')
column_prefix = 'ind-lstm-'
config.parameter.put('feature.ind.model.lstm-with-embedding.columns.prefix', column_prefix)
test_data_path = config.data.path('test_ind_uniquified.csv')
result_data_path = config.data.path('predict-ind-lstm-with-embedding')

ID, x, _ = reader_csv(test_data_path)
result = pd.DataFrame()

for i in range(n_estimator):
    print('加载模型{0}/{1}'.format(i+1, n_estimator))
    model = load_model(config.parameter.get('feature.ind.model.lstm-with-embedding.runtime.'+str(i)))
    _y = model.predict(x, verbose=True)
    _y = _y.reshape([-1])
    result[column_prefix+str(i)] = _y
result['id'] = ID
result.to_csv(result_data_path, index=False)


