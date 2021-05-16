from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator
import numpy as np
import pandas

class CustomClassifier(BaseEstimator):
    """Dummy detector which ouputs 1-second long steps every two seconds."""

    def __init__(self):
    	self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    	self.clf = LogisticRegression(random_state=42)

    def fit(self, X, y):
        embeddings = self.model.encode(list(X['Text']), show_progress_bar=True)
        print(y.shape)
        self.clf.fit(embeddings, y)
        return self

    def predict(self, X):
        predict = self.clf.predict(self.model.encode(list(X['Text'])))
        enc=OneHotEncoder()
        return enc.fit_transform(predict.reshape(-1,1)).toarray()


def get_estimator():
    return CustomClassifier()
