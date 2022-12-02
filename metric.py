from sklearn import metrics
import numpy as np


def calc_f1(y_true, y_pred,is_sigmoid):
    if not is_sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

def calc_metrics(y_true, y_pred,is_sigmoid):
    if not is_sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    accuracy = metrics.accuracy_score(y_true, y_pred)
    macro_f1 = metrics.f1_score(y_true, y_pred, average="macro")
    precision = metrics.precision_score(y_true, y_pred, pos_label=1)
    recall = metrics.recall_score(y_true, y_pred,pos_label=1)
    # f1_score = metrics.f1_score(y_true, y_pred,pos_label=1)
    f1_score = 2*precision*recall/(precision+recall+1e-10)
    return accuracy,macro_f1,precision,recall,f1_score


