#!/usr/bin/env python
# -*-coding:utf8-*-
import numpy

from Set import Set
import numpy as np
from itertools import combinations
import sys
import matplotlib.pyplot as plt


class LS(Set):
    """
    A class to create a set of data for parzen prediction

    Attributes
    ----------
    features : np.array
        Array corresponding to all the variables in the dataset
    labels : np.array
        Array corresponding to all the targets in the dataset
    features_names : list or None
        list corresponding to the name of each variable.
    epoch : int
        Max number of epoch
    min_epoch : int
        Minimal number of epoch
    sliding_window : int
        Number of hyperplan stock and compare to evaluate covergence
    stop : int
        Number of following epoch with convergence to stop
    hyperplan_type : str ['one vs one', 'one vs all']
        Hyperplan type

    Methods
    -------
    """
    def __init__(self, features, labels, features_names=None, epoch=50, min_epoch=10, sliding_window=20, stop=50,
                 hyperplan_type='one vs one'):
        """
        :param features: Array corresponding to all the variables in the dataset
        :param labels: Array corresponding to all the targets in the dataset
        :param features_names: list corresponding to the name of each variable.
        :param epoch: Max number of epoch
        :param min_epoch: Minimal number of epoch
        :param sliding_window: Number of hyperplan stock and compare to evaluate covergence
        :param stop: Number of following epoch with convergence to stop
        :param hyperplan_type: Hyperplan type ['one vs one', 'one vs all']

        :type features : np.array
        :type labels : np.array
        :type features_names : list or None
        :type epoch : int
        :type min_epoch : int
        :type sliding_window : int
        :type stop : int
        :type hyperplan_type : str
        """
        Set.__init__(self, features, labels, features_names)
        self.hyperplan_type = hyperplan_type
        self.epoch, self.min_epoch, self.sliding_window, self.stop = epoch, min_epoch, sliding_window, stop
        self.hyperplan = self._hyperplan()

    def _transform(self, target):
        """
        Allow to create a matrix with only the features corresponding to a target

        :param target : target name from the data

        :return
        """
        index = np.where(self.labels == target)[0]
        return np.concatenate((self.features[index], np.ones((np.shape(index)[0], 1))), axis=1)

    def _conv_slidding_windows(self):
        """

        """
        if len(np.unique(self.__best[:, -1])) == 1:
            return True
        else:
            return False

    def _new_best(self):
        """

        :return:
        """
        new_best = np.delete(self.__best, 0, axis=0)
        return np.concatenate((new_best, np.reshape(self.__best_epoch, (1, self.__best_epoch.shape[0]))), axis=0)

    def _calc_hyperplan_epoch(self, y):
        """
        Compute the hyperplan for one epoch
        """
        continu = True
        for elem in y:
            prod = np.dot(self.__hyperplan.T, elem)
            if prod <= 0:
                if self.__count_gc >= self.__best_epoch[-1]:
                    self.__best_epoch[-1] = self.__count_gc
                    self.__best_epoch[:-1] = self.__hyperplan
                self.__count_gc = 0
                self.__hyperplan += elem
            else:
                self.__count_gc += 1
                if self.__count_gc == self.stop:
                    self.__best_epoch[-1] = self.__count_gc
                    self.__best_epoch[:-1] = self.__hyperplan
                    continu = False
                    break
        return continu

    def _calc_hyperplan(self, y):
        """
        Compute the hyperplan for one epoch
        """
        self.__e, self.__count_gc = (0, 0)
        self.__best = np.zeros((self.sliding_window, (np.shape(y)[1] + 1)))
        self.__best_epoch = np.zeros((np.shape(y)[1] + 1))
        self.__hyperplan = np.zeros((np.shape(y)[1]))
        while self.__e < self.epoch:
            if self._calc_hyperplan_epoch(y):  # unmet stop condition
                self.__best = self._new_best()
                if self.__e > self.min_epoch and self._conv_slidding_windows():
                    break
            self.__e += 1
        return self.__best[-1, :-1]

    def _hyperplan_one(self):
        """
        Allow to compute the hyperplan one vs one from a Set
        """
        hyperplan_list = []
        for target_comb in combinations(self.targets, 2):  # two-by-two combination of all the target
            transform_1 = self._transform(target_comb[0])
            transform_2 = -self._transform(target_comb[1])
            hyperplan_list.append(self._calc_hyperplan(np.concatenate((transform_1, transform_2), axis=0)))
        return np.array(hyperplan_list)

    def _hyperplan_all(self):
        """
        Allow to compute the hyperplan one vs all from a Set
        """
        hyperplan_list = []
        for target in self.targets:
            transform_1 = self._transform(target)
            transform_2 = np.concatenate(np.array([-self._transform(t) for t in self.targets[self.targets != target]]))
            hyperplan_list.append(self._calc_hyperplan(np.concatenate((transform_1, transform_2), axis=0)))
        return np.array(hyperplan_list)

    def _hyperplan(self):
        """
        Allow to compute the hyperplan from a Set
        """
        if self.hyperplan_type == 'one vs one':
            return self._hyperplan_one()
        elif self.hyperplan_type == 'one vs all':
            return self._hyperplan_all()
        else:
            print("Error : Hyperplan compute type expected is one vs one or one vs all")
            sys.exit()

    def _classify(self, inX):
        prod = np.dot(inX, self.hyperplan.T[:-1, :]) + self.hyperplan.T[-1, :]
        combi_pred = np.where(prod > 0, 0, np.where(prod < 0, 1, 2))
        if self.hyperplan_type == 'one vs one':
            targets_count = np.array([combi[pred] for pred, combi in zip(combi_pred, combinations(self.targets, 2))])
            targets_count = self._count_targets(targets_count)
        elif self.hyperplan_type == 'one vs all':
            targets_count = np.zeros(self.targets.shape)
            for i in range(0, len(combi_pred)):
                if combi_pred[i] == 0:
                    targets_count[i] += 1
                elif combi_pred[i] == 1:
                    targets_count = np.array([x + 1 if j != i else x for j, x in enumerate(targets_count)])
        else:
            print("Error : Hyperplan compute type expected is one vs one or one vs all")
            sys.exit()
        return np.array([self.targets[targets_count.argsort()[::-1]],
                         targets_count[targets_count.argsort()[::-1]]])

    def predict(self, test_set, n_predict=3):
        predictions = []
        for sample in test_set:
            predictions.append(self._classify(sample)[:, :n_predict])
        self.preds = np.stack(predictions)
        return np.stack(predictions)

    def _train(self, new_train):
        return LS(new_train[:, :-1], new_train[:, -1], epoch=self.epoch, min_epoch=self.min_epoch,
                  sliding_window=self.sliding_window, stop=self.stop, hyperplan_type=self.hyperplan_type)

    def _plot_hyperplan_one_vs_one(self, title=None, fig_size=None):
        x = np.linspace(np.min(self.features), np.max(self.features), 10)
        for idx_comb, target_comb in enumerate(combinations(self.targets, 2)):
            self.plot_set(title, fig_size, close=False)
            y = (self.hyperplan[idx_comb][0]*x+self.hyperplan[idx_comb][2])/-self.hyperplan[idx_comb][1]
            plt.plot(x, y)
            plt.xlim(np.min(self.features[:, 0]), np.max(self.features[:, 0]))
            plt.ylim(np.min(self.features[:, 1]), np.max(self.features[:, 1]))
            if title is not None:
                plt.title(title + "\n Hyperplan between class {0} and {1}".format(str(target_comb[0]),
                                                                                  str(target_comb[1])))
            else:
                plt.title("Hyperplan between class {0} and {1}".format(str(target_comb[0]), str(target_comb[1])))
            plt.show()

    def _plot_hyperplan_one_vs_all(self, title=None, fig_size=None):
        x = np.linspace(np.min(self.features), np.max(self.features), 10)
        for idx, target in enumerate(self.targets):
            self.plot_set(title, fig_size, close=False)
            y = (self.hyperplan[idx][0]*x+self.hyperplan[idx][2])/(-self.hyperplan[idx][1])
            plt.plot(x, y)
            plt.xlim(np.min(self.features[:, 0]), np.max(self.features[:, 0]))
            plt.ylim(np.min(self.features[:, 1]), np.max(self.features[:, 1]))
            if title is not None:
                plt.title(title + "\n Hyperplan between class {0} and all".format(str(target)))
            else:
                plt.title("Hyperplan between class {0} and all".format(str(target)))
            plt.show()

    def plot_hyperplan(self, title=None, fig_size=None):
        if self.hyperplan_type == 'one vs one':
            self._plot_hyperplan_one_vs_one(title, fig_size)
        elif self.hyperplan_type == 'one vs all':
            self._plot_hyperplan_one_vs_all(title, fig_size)


if __name__ == '__main__':
    import os
    from sklearn.linear_model import Perceptron
    from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
    from sklearn.metrics import accuracy_score

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
    train_set = Set(train_data[:, 1:], train_data[:, 0])
    test_set = Set(test_data[:, 1:], test_data[:, 0])
    # plot3d(train_data)
    #
    train_ls_one = LS(features=train_set.features, labels=train_set.labels, epoch=1000, min_epoch=100,
                      sliding_window=10, stop=200)
    train_ls_one.predict(test_set=test_set.features, n_predict=2)
    preds_one = train_ls_one.predict(test_set=test_set.features, n_predict=2)
    acc, rej = train_ls_one.accuracy(test_set.labels, reject=True)
    print('Our Linear separation Accuracy with one vs one: {0}, rej {1}'.format(acc, rej))
    print(train_ls_one.confusion_matrix(test_set.labels, p_sum=False, reject=True))
    train_ls_one.plot_good_pred(test_set, title="Good pred", fig_size=[14, 9], reject=True)
    # 
    clf = OneVsOneClassifier(Perceptron(max_iter=1000))
    clf.fit(train_ls_one.features, train_ls_one.labels)
    p = clf.predict(test_set.features)
    print('Sklearn Perceptron model Accuracy: {}'.format(accuracy_score(test_set.labels, p)))

    # 
    index_5 = np.where(train_set.labels == 5)
    index_5_t = np.where(test_set.labels == 5)
    test_set_reduce = Set(features=np.delete(test_set.features, index_5_t, axis=0),
                          labels=np.delete(test_set.labels, index_5_t, axis=0))
    train_ls_all = LS(features=np.delete(train_set.features, index_5, axis=0),
                      labels=np.delete(train_set.labels, index_5, axis=0),
                      epoch=1000, min_epoch=500, sliding_window=400, stop=400, hyperplan_type='one vs all')
    preds_all = train_ls_all.predict(test_set=test_set_reduce.features, n_predict=2)
    acc2, rej2 = train_ls_all.accuracy(test_set_reduce.labels, reject=True)
    print('Our Linear separation Accuracy with one vs all: {0}, rej {1}'.format(acc2, rej2))
    print(train_ls_all.confusion_matrix(test_set_reduce.labels, p_sum=True, reject=True))
    train_ls_all.plot_good_pred(test_set_reduce, title="Good pred", fig_size=[14, 9], reject=True)
    #
    clf = OneVsRestClassifier(Perceptron(max_iter=1000))
    clf.fit(train_ls_all.features, train_ls_all.labels)
    p = clf.predict(test_set_reduce.features)
    print('Sklearn Perceptron model Accuracy: {}'.format(accuracy_score(test_set_reduce.labels, p)))

