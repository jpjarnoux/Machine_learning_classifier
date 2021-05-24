#!/usr/bin/env python
# -*-coding:utf8-*-

from __future__ import annotations


from Set import Set, plot_accuracy
import numpy as np


def plot_best_h(train: Parzen, h_min: int, h_max: int, step=1, title=None, fig_size=None, verbosity=False):
    h_list = np.arange(h_min, h_max, step)
    acc_list = []
    rej_list = []
    for i in h_list:
        new_train = Parzen(train.features, train.labels, h=i, seed=train.seed)
        acc, rej = new_train.cross_validation(n_fold=5, reject=True)
        acc_list.append(acc)
        rej_list.append(rej)
        if verbosity:
            print("Our Parzen model Accuracy with {0} and hyperparameter at {1}: {2} with a reject rate {3}".format(
                train.seed, i, acc, rej))
    plot_accuracy(h_list, np.stack(acc_list, axis=1), np.stack(rej_list, axis=1), title=title, fig_size=fig_size)


class Parzen(Set):
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
        h: int
            hyperparameter
        seed: str [majority, unanimity]
            type of seed for classification

        Methods
        -------
        """
    def __init__(self, features, labels, features_names=None, h=1, seed='gaussien'):
        """
        Initiate Parzen class

        :param features: Array corresponding to all the variables in the dataset
        :param labels: Array corresponding to all the targets in the dataset
        :param features_names: list corresponding to the name of each variable.
        :param h: hyperparameter
        :param seed: type of seed [gaussien, uniform]

        :type features: np.array
        :type labels: np.array
        :type features_names: list or None
        :type h: int
        :type seed: str
        """
        Set.__init__(self, features, labels, features_names)
        self.h = h
        self.seed = seed

    def _uniforme(self, dist):
        """
        Allow to classify sample features with an uniform seed

        :param dist: distance between sample's features and train features

        :type dist: np.array

        :return: np.array with count of each target
        """
        identify_labels = self.labels[np.where(dist < self.h)]
        # print(identify_labels)
        return self._count_targets(identify_labels)

    def _gaussien(self, dist):
        """
        Allow to classify sample features with an uniform gaussien

        :param dist: distance between sample's features and train features

        :type dist: np.array

        :return: np.array with count of each target
        """
        phy = (1/(2*np.pi*self.h))*np.exp((1/2) * dist / self.h)
        identify_labels = self.labels[np.where(phy < self.h)]
        # print(identify_labels)
        return self._count_targets(identify_labels)

    def _classify(self, inX):
        """
        Allow to classify sample features with an different seed

        :param inX: sample features

        :type inX: np.array

        :return: np.array with count of each target
        """
        eu_dist = np.sqrt(((self.features - inX) ** 2).sum(axis=1))
        if self.seed == 'uniform':
            return self._uniforme(eu_dist) / self.labels_counts
        elif self.seed == 'gaussien':
            return self._gaussien(eu_dist) / self.labels_counts
        else:
            print("Error : Seed expected method is uniform or gaussien")
            return 0

    def _train(self, new_train):
        return Parzen(new_train[:, :-1], new_train[:, -1], features_names=self.features_names, h=self.h, seed=self.seed)

    def _predict(self, classify, n_preds=3):
        """
        Allow to predict target with a parzen classication

        :param classify: parzen classication
        :param n_preds: number of predicted targets

        :type classify: np.array
        :type n_preds: int
        :return: predicted targets and count of target predicted
        """
        keep_idx = classify.argsort()[::-1][:n_preds]
        return np.vstack([self.targets[keep_idx], classify[keep_idx]])

    def predict(self, test_set: Set, n_predict=3):
        """
        Allow to predict target with a parzen classication

        :param test_set: test
        :param n_predict: number of predicted targets

        :type test_set: Set
        :type n_predict: int

        :return: predicted targets and count of target predicted for all samples in test
        """
        predictions = []
        # Loop through all samples, predict the class labels and store the results
        for sample in test_set:
            predictions.append(self._predict(classify=self._classify(sample), n_preds=n_predict))
        self.preds = np.stack(predictions)
        return np.stack(predictions)


if __name__ == '__main__':
    import os
    import pandas as pd
    from sklearn.neighbors import KernelDensity
    from sklearn.metrics import accuracy_score

    def read_file(filepath, delimiter=' '):
        contents = []
        with open(filepath) as f:
            for line in f:
                contents.append(line.strip().split(delimiter))

        return np.array(contents).astype(np.float64)

    def sk_accuracy(train: Parzen, test: Set, seed='gaussian'):
        preds = []
        for target in train.targets:
            clf_uni = KernelDensity(bandwidth=1, kernel=seed, metric='euclidean')
            index = np.where(train.labels == target)[0]
            clf_uni.fit(train.features[index, :], train.labels[index])
            # score_samples() returns the log-likelihood of the samples
            preds.append(clf_uni.score_samples(test.features))
        predictions = np.array([train.targets[pred_idx] for pred_idx in
                                pd.DataFrame(np.stack(preds, axis=1)).idxmax(axis=1).to_numpy()])
        print('Sklearn KNN model Accuracy: {}'.format(accuracy_score(test.labels, predictions)))


    data_dir = os.path.abspath("Data")
    print(data_dir)
    train_data = read_file(os.path.join(data_dir, "data_tp1_app.txt"))
    test_data = read_file(os.path.join(data_dir, "data_tp3_dec.txt"))
    train_set = Set(train_data[:, 1:], train_data[:, 0])
    test_set = Set(test_data[:, 1:], test_data[:, 0])
    # plot3d(train_data)

    train_parzen_uni = Parzen(features=train_set.features, labels=train_set.labels, h=10, seed='uniform')
    preds_uni = train_parzen_uni.predict(test_set=test_set.features, n_predict=4)
    acc, rej = train_parzen_uni.accuracy(test_set.labels, reject=True)
    print('Our Parzen model Accuracy with uniform: {0}, rej {1}'.format(acc, rej))
    print(train_parzen_uni.confusion_matrix(test_set.labels, p_sum=True, reject=True))
    train_parzen_uni.plot_good_pred(test_set, title="Good pred", fig_size=[14, 9], reject=True)

    sk_accuracy(train_parzen_uni, test_set, seed='linear')

    train_parzen_gau = Parzen(features=train_set.features, labels=train_set.labels, h=1, seed='gaussien')
    preds_gau = train_parzen_gau.predict(test_set=test_set.features, n_predict=3)
    acc, rej = train_parzen_gau.accuracy(test_set.labels, reject=True)
    print('Our Parzen model Accuracy with gaussien: {0}, rej {1}'.format(acc, rej))
    print(train_parzen_gau.confusion_matrix(test_set.labels, p_sum=False, norm=True, reject=True))
    train_parzen_gau.plot_good_pred(test_set, title="Good pred", fig_size=[14, 9], reject=True)

    sk_accuracy(train_parzen_uni, test_set)

    plot_best_h(train_parzen_uni, h_min=2, h_max=19)
