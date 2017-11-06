import numpy as np
import tensorflow as tf
import keras.backend as K
import keras
from sklearn.metrics import roc_auc_score


def gini(actual, pred, cmpcol=0, sortcol=1):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)


def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score


def jacek_auc(y_true, y_pred):
    score, up_opt = tf.metrics.auc(y_true, y_pred)
    # score, up_opt = tf.contrib.metrics.streaming_auc(y_pred, y_true)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score


class gini_callback(keras.callbacks.Callback):
    def __init__(self):
        # print("self vars: ",vars(self))  #uncomment and discover some things =)
        super().__init__()

    def on_train_begin(self, logs={}):
        print("Hi! on_train_begin() , logs shape:", np.shape(logs))
        # print("self vars: ",vars(self))  #uncomment and discover some things =)
        return

    def on_train_end(self, logs={}):
        print("Hi! on_train_end() , logs shape:", np.shape(logs))
        # print("self vars: ",vars(self))  #uncomment and discover some things =)
        return

    def on_epoch_begin(self, epoch, logs={}):
        print("Hi! on_epoch_begin() , epoch=", epoch, ",logs shape:", np.shape(logs))
        # print("self vars: ",vars(self))  #uncomment and discover some things =)
        return

    def on_epoch_end(self, epoch, logs={}):
        print("Hi! on_epoch_end() , epoch=", epoch, ",logs shape:", np.shape(logs))
        # print("self vars: ",vars(self))  #uncomment and discover some things =)

        print("    GINI Callback:")
        if (self.validation_data):
            print('        validation_data.inputs:  ', np.shape(self.validation_data[0]))
            print('        validation_data.targets: ', np.shape(self.validation_data[1]))
            # print("        roc_auc_score(y_real,y_hat): ",
            #       roc_auc_score(self.validation_data[1], self.model.predict(self.validation_data[0])))
            print("        gini_normalized(y_real,y_hat): ",
                  gini_normalized(self.validation_data[1], self.model.predict(self.validation_data[0])),
                  "/roc_auc_scores*2-1=",
                  roc_auc_score(self.validation_data[1], self.model.predict(self.validation_data[0])) * 2 - 1)
        return

    def on_batch_begin(self, batch, logs={}):
        if (batch != 0):
            print("")
        print("Hi! on_batch_begin() , batch=", batch, ",logs shape:", np.shape(logs))
        # print("self vars: ",vars(self))  #uncomment and discover some things =)
        return

    def on_batch_end(self, batch, logs={}):
        print("Hi! on_batch_end() , batch=", batch, ",logs shape:", np.shape(logs))
        # print("self vars: ",vars(self))  #uncomment and discover some things =)
        return
