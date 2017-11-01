from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import concatenate
from keras.layers.embeddings import Embedding


class EmbeddedLSTM(object):

    class Config(object):
        def __init__(self):
            self.embedding_word_number = None
            self.embeding_vector_length = None
            self.x_int_columns = None
            self.x_float_columns = None
            self.dropout = 0.2
            self.layer_dense = [16]

    def __init__(self):
        self._config = EmbeddedLSTM.Config()
        self._model = None

    def build(self):
        assert self._config is not None, "模型未配置, 不能初始化"

        config = self._config

        x_int = Input(shape=(len(config.x_int_columns),), dtype='int32', name='x_int')
        x_int_vec = Embedding(
            name='embedding',
            input_dim=config.embedding_word_number,
            output_dim=config.embeding_vector_length,
            input_length=len(config.x_int_columns)
        )(x_int)

        if config.dropout > 0:
            x_int_vec = Dropout(config.dropout)(x_int_vec)

        x_int_lstm_embedded_out = LSTM(units=32, name='lstm-embedded')(x_int_vec)

        if config.dropout > 0:
            x_int_lstm_embedded_out = Dropout(config.dropout)(x_int_lstm_embedded_out)

        y_aux = Dense(1, activation='sigmoid', name='y_aux')(x_int_lstm_embedded_out)

        # 浮点特征
        x_float = Input(shape=(len(config.x_float_columns),), name='x_float')
        x = concatenate([y_aux, x_float])

        # x_lstm_out = LSTM(64, name='lstm')(Reshape((-1, 32+len(config.x_float_columns)))(x))
        # if config.dropout > 0:
        #     x_lstm_out = Dropout(config.dropout)(x_lstm_out)

        x_dense = x
        for i, dim in enumerate(config.layer_dense):
            x_dense = Dense(64, activation='sigmoid', name='dense_'+str(i))(x_dense)
            if config.dropout > 0:
                x_dense = Dropout(config.dropout)(x_dense)

        # 输出
        y = Dense(1, activation='sigmoid', name='y')(x_dense)

        self._model = Model(inputs=[x_int, x_float], outputs=[y, y_aux])

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
    config = EmbeddedLSTM.Config()
    config.x_int_columns = ['0', '1', '2', '3', '4', '5']
    config.x_float_columns = ['6', '7', '8']
    config.embedding_word_number = 600
    config.embeding_vector_length = 32

    lstm = EmbeddedLSTM()
    lstm.config = config
    lstm.build()
    lstm.model.summary()






