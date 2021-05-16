from sklearn.dummy import DummyClassifier


def get_estimator():
    return DummyClassifier(strategy="most_frequent")