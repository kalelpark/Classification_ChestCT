from sklearn import metrics
import numpy as np


def get_accuracy(y_true, y_pred):
    score = metrics.accuracy_score(y_true, y_pred)
    return score

def get_f1_score(y_true, y_pred):
    score = metrics.f1_score( y_true, y_pred, 
                              average='macro')
    return score
