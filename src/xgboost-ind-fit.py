"""
XGBOOST 模型
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from src.pipeline.reader import reader_csv
import xgboost as xgb
from setup import config

########################################################################################################################
# 配置

data_train_ind_path = config.data.path('train_ps_ind.csv')
feature_ind_model_xgboost_path = config.runtime.path('feature_ind_model_xgboost')
config.parameter.put('feature.ind.model.xgboost.runtime', feature_ind_model_xgboost_path)

########################################################################################################################
# 构造数据

_, x, y = reader_csv(data_train_ind_path)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

# ########################################################################################################################
# 构建模型

print("#"*120)
print("构建模型")
xgb_model = xgb.XGBClassifier(
    objective= 'binary:logistic',
    scale_pos_weight=1/0.034,
    max_depth=5,
    min_child_weight=1,
    nthread=8,
)

# ########################################################################################################################
print("#"*120)
print("训练")

rng = np.random.RandomState(31337)
kf = KFold(n_splits=4, shuffle=True, random_state=rng)

for i, (train_index, test_index) in enumerate(kf.split(x_train, y_train)):
    xgb_model.fit(x_train[train_index], y_train[train_index])

    _y = xgb_model.predict(x_train[test_index])
    y = y_train[test_index]

    confuse = confusion_matrix(y, _y)
    print("第{0}次, CV混淆矩阵 = ".format(i))
    print(confuse)

    acc = precision_recall_fscore_support(xgb_model.predict(x_test), y_test)

    print("第{0}次, 测试集准确率 = ".format(i))
    print(acc)
    # xgb.save_model(xgb_model, feature_ind_model_xgboost_path)

# ########################################################################################################################
print("#"*120)
print("预测")

test_data_path = config.data.path('test_ps_ind.csv')

ID, x, _ = reader_csv(test_data_path)
_y = xgb_model.predict(x)

result = pd.DataFrame({'id': ID, 'target': _y})

rv_path = config.data.path('predict-ind-xgboost.csv')

result.to_csv(rv_path, index=False)


# predictions = xgb_model.predict(x)
#
# from sklearn.metrics import confusion_matrix
#
# confuse = confusion_matrix(y, predictions)
#
# print(confuse)
#
# import numpy as np
# rng = np.random.RandomState(31337)
# kf = KFold(n_splits=10, shuffle=True, random_state=rng)
#
# # for train_index, test_index in kf.split(x, y):

