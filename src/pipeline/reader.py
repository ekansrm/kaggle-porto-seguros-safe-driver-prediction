"""
封装文件的读取
目标是, 无论数据在本地, 还是oss, db, 生产的迭代器都无感知
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Iterable


def reader_csv(path: str) -> (np.array, np.array):
    df = pd.read_csv(path)
    columns = list(df.columns)
    if 'id' in columns:
        df.drop(labels=['id'], axis=1, inplace=True)

    y = df['target']
    x = df.drop(labels=['target'], axis=1)
    return x.values, y.values


def generate(sample: Iterable, sample_name: str, batch_size: int, name_scope: str = None):
    """ 返回数据的迭代器
  Args:
    :param sample_name:
    :param sample:
    :param batch_size: batch的数目
    :param name_scope:
  Returns:
      两个迭代器
  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
    with tf.name_scope(name_scope, "Producer", [sample, batch_size]):
        sample = tf.convert_to_tensor(sample, name=sample_name, dtype=tf.int32)

        # 获取样本的个数
        sample_num = tf.size(y)
        batch_num = sample_num // batch_size
        assertion = tf.assert_positive(
            batch_num,
            message="batch_num == 0, decrease batch_size"
        )
        with tf.control_dependencies([assertion]):
            batch_num = tf.identity(batch_num, name="batch_num")

        sample_dim = [d.value for d in sample.shape[1:]]

        # 数据按批次大小取整, 并reshape
        if len(sample_dim) == 0:
            sample = tf.reshape(sample[0: batch_num * batch_size], [batch_num, batch_size])
        else:
            sample = tf.reshape(sample[0: batch_num * batch_size], [batch_num, batch_size, *sample_dim])

        batch_idx = tf.train.range_input_producer(batch_num, shuffle=False).dequeue()

        sample_batch = tf.strided_slice(sample, [batch_idx], [(batch_idx + 1)])[0]

        return sample_batch


if __name__ == '__main__':
    # from setup import config
    # data_path = config.data.path('train_ps_ind.csv')
    # x, y = reader_csv(data_path)

    # 测试迭代器
    x = np.asarray(list(range(0, 1024*8)), dtype=np.int32).reshape([1024, 8])
    y = np.asarray(list(range(0, 1024)), dtype=np.int32)

    x_batch = generate(sample=x, sample_name="x", batch_size=7, name_scope="123")
    y_batch = generate(sample=y, sample_name="y", batch_size=7, name_scope="123")

    with tf.Session() as session:
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(session, coord=coord)
        try:
            for i in range(0, 16):
                xval, yval = session.run([x_batch, y_batch])
                print(xval)
                print(yval)
        finally:
            coord.request_stop()
            coord.join()
