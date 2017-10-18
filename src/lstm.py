"""
LSTM 模型

"""
import tensorflow as tf
import numpy as np
from enum import Enum
from src.pipeline.reader import generate


class LSTM(object):
    """
    LSTM 模型, 包括 LSTM 单元 和全连接层

    embeding: (vocab_size, hidden_size)

    lstm: (hidden_layer_size, hidden_layer_size)

    full-connect: (hidden_layer_size, output_size) -> 期待输出

    """

    class Config(object):
        """
        模型配置:
        配置是在build之前就得配置好的
        """
        class Mode(Enum):
            BASIC = "basic"
            CUDNN = "cudnn"
            BLOCK = "block"

        dtype = tf.float32              # 数据类型
        mode = Mode.CUDNN               # 模型类型

        var_init_scale = 0.1      # embedding 初始权重
        vocab_size = 10000

        lstm_nums_layer = 2       # 层数
        lstm_nums_units = 200    # 隐藏层大小

        x_dim = None
        y_dim = None

        max_epoch = 4
        max_max_epoch = 13

        batch_size = 20
        num_steps = 20

        # 训练
        keep_prob = 1.0                 # 保持概率
        learning_rate = 1.0             # 学习率
        learning_rate_decay = 0.5       # 学习率衰减
        max_grad_norm = 5               # 最大梯度

        is_training = False

    @staticmethod
    def with_prefix(prefix, name):
        """添加前缀."""
        return "/".join((prefix, name))

    def __init__(self):
        """
        """
        self._is_training = False
        self._batch_size = None
        self._config = LSTM.Config()

        self._name = None

        self._rnn_params = None
        self._cell = None

        # 损失函数
        self._cost = None

        # 状态
        self._initial_state_name = None
        self._final_state_name = None
        self._initial_state = None
        self._final_state = None

        # 学习率
        self._lr = None
        self._new_lr = None
        self._op_lr_update = None

        # 训练操作子
        self._op_train = None

        # 输入和输出
        self._x = None
        self._y = None

    ####################################################################################################################
    # 模型配置

    @property
    def is_training(self):
        return self._is_training

    @is_training.setter
    def is_training(self, is_training: bool):
        self._is_training = is_training

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, _config: Config):
        assert isinstance(_config, LSTM.Config), "配置类型错误"
        # 只更新变化了的参数
        self._config = _config

    @config.getter
    def get_config(self)-> Config:
        return self._config

    def check(self):
        assert self._config is not None, "模型未配置"

    ####################################################################################################################
    # 模型构建

    def _build_rnn_graph(self):
        if self._config.mode == LSTM.Config.Mode.CUDNN:
            return self._build_rnn_graph_cudnn()
        else:
            return self._build_rnn_graph_lstm()

    def _build_rnn_graph_cudnn(self):

        """Build the inference graph using CUDNN cell.
        输出:
            outputs: 样本对应一个 长度为 lstm_nums_units 的输出
            state_tuple: lstm_nums_units, lstm_nums_units
        """
        inputs = tf.transpose(self._x, [1, 0, 2])
        self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=self._config.lstm_nums_layer,
            num_units=self._config.lstm_nums_units,
            input_size=self._config.lstm_nums_units,
            dropout=1 - self._config.keep_prob if self._is_training else 0)
        params_size_t = self._cell.params_size()
        self._rnn_params = tf.get_variable(
            name="lstm_params",
            initializer=tf.random_uniform([params_size_t], -self._config.var_init_scale, self._config.var_init_scale),
            validate_shape=False
        )
        c = tf.zeros([self._config.lstm_nums_layer, self._config.batch_size, self._config.lstm_nums_units],
                     tf.float32)
        h = tf.zeros([self._config.lstm_nums_layer, self._config.batch_size, self._config.lstm_nums_units],
                     tf.float32)
        self._initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
        outputs, h, c = self._cell(inputs, h, c, self._rnn_params, self._is_training)
        outputs = tf.transpose(outputs, [1, 0, 2])
        return outputs, (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)

    def _build_rnn_graph_lstm(self):
        """
        使用标准的 LSTM 单元构造流图
        :return:
        """
        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.

        # 构造一个单元, 可以理解为一层
        with tf.variable_scope("RNN"):
            lstm_cell = self._make_lstm_cell(self._config.lstm_nums_units)
            lstm_cells = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell] * self._config.lstm_nums_layer
            )
        print([o.name for o in tf.get_default_graph().get_operations()])

        print(lstm_cells.output_size)
        print(lstm_cells.state_size)

        self._initial_state = lstm_cells.zero_state(self._config.batch_size, self.config.dtype)
        state = self._initial_state

        # 讲输入输入 LSTM的cell, 然后获取输出和最后状态
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(self._config.num_steps):
                if time_step > 0:  # 如果不是第一次, 就把当前变量空间设置为 reuse, 因为是要共享的
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = lstm_cells(self._x[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(values=outputs, axis=1), [-1, self._config.lstm_nums_units])

        return output, state

    def _make_lstm_cell(self, num_units):
        """
        根据配置, 生产对应的 LSTM 单元. BASIC|BLOCK
        :return:
        """
        if self._config.mode == LSTM.Config.Mode.BASIC:
            cell = tf.contrib.rnn.BasicLSTMCell(
                num_units=num_units,
                forget_bias=0.0,
                state_is_tuple=True,    # If True, accepted and returned states are 2-tuples of the c_state and m_state.
                # If False, they are concatenated along the column axis. The latter behavior
                # will soon be deprecated.

                reuse=not self._is_training

            )
        elif self._config.mode == LSTM.Config.Mode.BLOCK:
            cell = tf.contrib.rnn.LSTMBlockCell(
                num_units=self._config.lstm_nums_units,
                forget_bias=0.0,
            )
        else:
            raise ValueError("RNN 模式 'rnn_mode' 不支持 '%s' ", self._config.mode)

        if self._is_training and self._config.keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(
                cell=cell,
                output_keep_prob=self._config.keep_prob
            )
        print([o.name for o in tf.get_default_graph().get_operations()])
        return cell

    def build(self):

        self.check()

        initializer = tf.random_uniform_initializer(-self._config.var_init_scale, self._config.var_init_scale)
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            self._x = tf.placeholder(
                name='x',
                dtype=self._config.dtype,
                shape=[self._config.batch_size, self._config.num_steps, self._config.lstm_nums_units]
            )
            self._y = tf.placeholder(
                name='y',
                dtype=tf.int32,
                shape=[self._config.batch_size, self._config.y_dim]
            )

        # print([o.name for o in tf.get_default_graph().get_operations()])

        # 构建RNN
        # RNN 输出的张量是 [batch_size, nums_step, lstm_nums_unit]
        f_lstm, state = self._build_rnn_graph()
        # 取最后一步的RNN输出
        f_lstm = f_lstm[:, -1, :]

        # 构建全连接层作为输出层
        softmax_w = tf.get_variable(
            name="softmax_w",
            shape=[self._config.lstm_nums_units, self._config.y_dim],
            dtype=self._config.dtype
        )
        softmax_b = tf.get_variable(
            name="softmax_b",
            shape=[self._config.y_dim],
            dtype=self._config.dtype
        )
        f_softmax = tf.nn.softmax(tf.nn.xw_plus_b(f_lstm, softmax_w, softmax_b))
        print(f_softmax.shape)
        # 使用tensorflow的函数计算序列交叉熵
        # loss = tf.contrib.seq2seq.sequence_loss(
        #     f_softmax,
        #     self._y,
        #     tf.ones([self._config.batch_size, self._config.y_dim], dtype=self._config.dtype),
        #     average_across_timesteps=False,
        #     average_across_batch=True
        # )
        # print(loss)
        # cross_entopy = \
        #     tf.reduce_sum(self._y, -tf.log(f_softmax)) / tf.cast(self._config.batch_size, dtype=self._config.dtype)
        # print(cross_entopy)
        cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self._y, [-1]),
            logits=f_lstm
        ))
        print(cross_entropy_mean)

        self._cost = cross_entropy_mean

        # 更新状态
        self._final_state = state

        # 只在训练模型时定义反向传播操作
        if not self._is_training:
            return

        # 梯度裁剪
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self._cost, tvars),
            self._config.max_grad_norm
        )

        # 定义训练操作
        self._lr = tf.Variable(0.0, trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(self._lr, name="optimizer")

        self._op_train = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step()
        )

        # 动态更新学习率
        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")

        self._op_lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._op_lr_update, feed_dict={self._new_lr: lr_value})

    # def export_ops(self, name):
    #     """Exports ops to collections."""
    #     self._name = name
    #     ops = {LSTM.with_prefix(self._name, "cost"): self._cost}
    #     if self._is_training:
    #         ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._op_lr_update)
    #         if self._rnn_params:
    #             ops.update(rnn_params=self._rnn_params)
    #     for name, op in ops.items():
    #         tf.add_to_collection(name, op)
    #     self._initial_state_name = LSTM.with_prefix(self._name, "initial")
    #     self._final_state_name = LSTM.with_prefix(self._name, "final")
    #     util.export_state_tuples(self._initial_state, self._initial_state_name)
    #     util.export_state_tuples(self._final_state, self._final_state_name)
    #
    # def import_ops(self):
    #     """Imports ops from collections."""
    #     if self._is_training:
    #         self._op_train = tf.get_collection_ref("train_op")[0]
    #         self._lr = tf.get_collection_ref("lr")[0]
    #         self._new_lr = tf.get_collection_ref("new_lr")[0]
    #         self._op_lr_update = tf.get_collection_ref("lr_update")[0]
    #         rnn_params = tf.get_collection_ref("rnn_params")
    #         if self._cell and rnn_params:
    #             params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
    #                 self._cell,
    #                 self._cell.params_to_canonical,
    #                 self._cell.canonical_to_params,
    #                 rnn_params,
    #                 base_variable_scope="Model/RNN")
    #             tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
    #     self._cost = tf.get_collection_ref(LSTM.with_prefix(self._name, "cost"))[0]
    #     num_replicas = FLAGS.num_gpus if self._name == "Train" else 1
    #     self._initial_state = util.import_state_tuples(
    #         self._initial_state, self._initial_state_name, num_replicas)
    #     self._final_state = util.import_state_tuples(
    #         self._final_state, self._final_state_name, num_replicas)

    ####################################################################################################################
    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state

    @property
    def initial_state_name(self):
        return self._initial_state_name

    @property
    def final_state_name(self):
        return self._final_state_name

    @property
    def cost(self):
        return self._cost

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, _x):
        self._x = _x

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, _y):
        self._y = _y

    @property
    def lr(self):
        return self._lr

    @property
    def op_train(self):
        return self._op_train

    @property
    def op_update_lr(self):
        return self._op_lr_update


if __name__ == '__main__':
    config = LSTM.Config()
    config.batch_size = 20
    config.num_steps = 8
    config.x_dim = 1
    config.y_dim = 1
    config.lstm_nums_units = 8
    config.is_training = True
    config.max_grad_norm = 13

    lstm = LSTM()
    lstm.config = config

    train_number = 1024*16

    x = np.asarray(list(range(0, train_number*8)), dtype=np.int32).reshape([train_number, 8])
    y = np.asarray(list(range(0, train_number)), dtype=np.int32)

    g2 = tf.Graph()
    with g2.as_default():
        x_batch = generate(sample=x, sample_name="x", batch_size=7, name_scope="train/input")
        y_batch = generate(sample=y, sample_name="y", batch_size=7, name_scope="train/input")
        lstm.x = x_batch
        lstm.y = y_batch
        lstm.is_training = True
        lstm.build()

    with tf.Session(graph=g2) as session:
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(session, coord=coord)
        try:
            for i in range(0, 16):
                op_train = session.run([lstm.op_train], feed_dict={})
                print(op_train)
        finally:
            coord.request_stop()
            coord.join()
