"""
IsolationForest
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from setup import config
from src.pipeline.reader import reader_csv

########################################################################################################################
# 配置

data_train_ind_path = config.data.path('train_ind.csv')
# feature_ind_model_xgboost_path = config.runtime.path('feature_ind_model_xgboost')
# config.parameter.put('feature.ind.model.xgboost.runtime', feature_ind_model_xgboost_path)


########################################################################################################################
# 构造数据

_, x, y = reader_csv(data_train_ind_path)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

# ########################################################################################################################
model = IsolationForest(
    n_estimators=200,
    n_jobs=-1,
    verbose=0
)


print("#"*120)
print("训练")

rng = np.random.RandomState(31337)
kf = KFold(n_splits=10, shuffle=True, random_state=rng)

for i, (train_index, test_index) in enumerate(kf.split(x_train, y_train)):
    model.fit(x_train[train_index], y_train[train_index])

    _y = model.predict(x_train[test_index])

    _y[(_y<0)] = 0

    print(np.unique(_y))
    y = y_train[test_index]

    confuse = confusion_matrix(y, _y)
    print("第{0}次, CV混淆矩阵 = ".format(i))
    print(confuse)

    acc = precision_recall_fscore_support(model.predict(x_test), y_test)

    print("第{0}次, 测试集准确率 = ".format(i))
    print(acc)
    # xgb.save_model(xgb_model, feature_ind_model_xgboost_path)

# ########################################################################################################################
