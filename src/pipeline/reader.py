"""
封装文件的读取
目标是, 无论数据在本地, 还是oss, db, 生产的迭代器都无感知
"""

import pandas as pd
import numpy as np
import tensorflow as tf


def reader_csv(path: str) -> (np.array, np.array):
    df = pd.read_csv(path)
    columns = list(df.columns)
    if 'id' in columns:
        df.drop(labels=['id'], axis=1, inplace=True)

    y = df['target']
    x = df.drop(labels=['target'], axis=1)
    return x.values, y.values


def generate(x: np.array, y: np.array, batch_size: int, num_steps, name: str = None):
    """ 返回数据的迭代器
  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).
  Returns:
      两个迭代器
  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
    with tf.name_scope(name, "Producer", [x, y, batch_size, num_steps]):
        x = tf.convert_to_tensor(x, name="x", dtype=tf.int32)
        y = tf.convert_to_tensor(y, name="y", dtype=tf.int32)

        data_len = tf.size(y)
        batch_len = data_len // batch_size

        x = tf.reshape(x[0: batch_size * batch_len, :], [batch_size, batch_len])
        y = tf.reshape(y[0: batch_size * batch_len], [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")

        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps],
                             [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1],
                             [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y


if __name__ == '__main__':
    data_path = '/Users/kami.wm/DataScience/data/kaggle-porto-seguros-safe-driver-prediction/train_ps_ind.csv'
    x, y = reader_csv(data_path)
    print(x, y)

