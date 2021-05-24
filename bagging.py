#!/usr/bin/env python
# -*-coding:utf8-*-

from Set import Set
import numpy as np
import pandas as pd


class Bagging(Set):
    def __init__(self, features, labels, features_names=None):
        Set.__init__(self, features, labels, features_names)
        self.models = []
        self.models_attr = []
        self.preds = []

    def add_models(self, model, times=1, random_state=None, **kargs):
        while times != 0:
            new_train = self._shuffle_set(random_state, replace=True)
            self.models_attr.append(kargs)
            self.models.append(model(new_train[:, :-1], new_train[:, -1], self.features_names, **kargs))
            times += -1

    def predict(self, test, n_predict):
        preds_list = []
        for model in self.models:
            model.predict(test, n_predict)
            if len(model.preds.shape) == 2:
                preds_list.append(model.preds)
            else:
                preds_list.append(model.preds[:, 0, :])
        bagging = pd.DataFrame(np.hstack(preds_list).T)
        self.preds = np.array([[bagging[column].value_counts().idxmax()] for column in bagging])

    def _train(self, new_train):
        new_bagging = Bagging(new_train[:, :-1], new_train[:, -1], features_names=self.features_names)
        for idx, model in enumerate(self.models):
            new_bagging.add_models(model=model.__class__, **self.models_attr[idx])
        return new_bagging


if __name__ == '__main__':
    import os
    from ls import LS
    from knn import KNN
    from parzen import Parzen


    def read_file(filepath, delimiter=' '):
        contents = []
        with open(filepath) as f:
            for line in f:
                contents.append(line.strip().split(delimiter))

        return np.array(contents).astype(np.float64)


    data_dir = os.path.abspath("Data")
    print(data_dir)
    train_data = read_file(os.path.join(data_dir, "data_tp1_app.txt"))
    test_data = read_file(os.path.join(data_dir, "data_tp1_dec.txt"))
    test_set = Set(test_data[:, 1:], test_data[:, 0])

    train_bagging = Bagging(train_data[:, 1:], train_data[:, 0])
    train_bagging.add_models(model=LS, times=4, epoch=100, min_epoch=20)
    train_bagging.add_models(model=KNN, times=6, k=4)
    train_bagging.add_models(model=Parzen, times=12, h=1, seed='gaussien')
    train_bagging.predict(test=test_set.features, n_predict=1)
    train_bagging.accuracy(test_set.labels, reject=False)
    train_bagging.confusion_matrix(test_set.labels, reject=False)


