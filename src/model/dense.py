from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import concatenate
from keras.layers.embeddings import Embedding
from keras import backend as K


class TotalDense(object):

    class Config(object):
        def __init__(self):
            self.x_dim = None
            self.y_dim = None
            self.dropout = 0.2
            self.dense = [16]

    def __init__(self):
        self._config = TotalDense.Config()
        self._model = None

    def build(self):
        assert self._config is not None, "模型未配置, 不能初始化"

        config = self._config

        x = Input(shape=(config.x_dim,), dtype='float32', name='x')

        x_dense = x
        for i, dim in enumerate(config.dense):
            x_dense = Dense(dim, activation='tanh', name='dense_'+str(i))(x_dense)
            if config.dropout > 0:
                x_dense = Dropout(config.dropout)(x_dense)

        # 输出
        y = Dense(config.y_dim, activation='sigmoid', name='y')(x_dense)

        self._model = Model(inputs=[x], outputs=[y])

    @property
    def model(self) -> Model:
        return self._model

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: Config):
        self._config = config


if __name__ == '__main__':
    config = TotalDense.Config()
    config.x_dim = 100
    config.y_dim = 1
    config.dense = [200, 100, 50]

    dense = TotalDense()
    dense.config = config
    dense.build()
    dense.model.summary()






