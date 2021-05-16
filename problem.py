 
import os
import pandas as pd
import rampwf as rw
from sklearn.metrics import f1_score
from rampwf.score_types.classifier_base import ClassifierBaseScoreType
from sklearn.model_selection import StratifiedShuffleSplit

class F1Weighted(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='f1_weighted', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        f1 = f1_score(y_true, y_pred, average="weighted")
        return f1
        
        
problem_title = 'Exploring Racism and Sexism in Social Media'
_target_column_name = 'target'
_prediction_label_names = [0, 1,2]
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
# An object implementing the workflow
workflow = rw.workflows.Estimator()


score_types = [
    F1Weighted()
]



def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name], axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)
