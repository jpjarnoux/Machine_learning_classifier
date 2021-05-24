#!/usr/bin/env python
# -*-coding:utf8-*-

from __future__ import annotations


from Set import Set, plot_accuracy
import numpy as np


def plot_best_k(train: KNN, k_min: int, k_max: int, step=1, title=None, fig_size=None, verbosity=False):
    k_list = range(k_min, k_max, step)
    acc_list = []
    rej_list = []
    for i in k_list:
        new_train = KNN(train.features, train.labels, k=i, vote=train.vote)
        acc, rej = new_train.cross_validation(n_fold=5, reject=True)
        acc_list.append(acc)
        rej_list.append(rej)
        if verbosity:
            print("{0}NN model Accuracy with majority vote : {1} with a reject rate {2}".format(i, acc, rej))
    plot_accuracy(k_list, np.stack(acc_list, axis=1), np.stack(rej_list, axis=1), title=title, fig_size=fig_size)


class KNN(Set):
    """
    A class to create a set of data for KNN prediction

    Attributes
    ----------
    features : np.array
        Array corresponding to all the variables in the dataset
    labels : np.array
        Array corresponding to all the targets in the dataset
    features_names : list or None
        list corresponding to the name of each variable.
    k: int
        number of neighbors
    vote: str [majority, unanimity]
        type of vote for prediction

    Methods
    -------
    """

    def __init__(self, features, labels, features_names=None, k=3, vote='majority'):
        """
        Initiate KNN class
        :param features: Array corresponding to all the variables in the dataset
        :param labels: Array corresponding to all the targets in the dataset
        :param features_names: list corresponding to the name of each variable.
        :param k: number of neighbors
        :param vote: type of vote for prediction

        :type features: np.array
        :type labels: np.array
        :type features_names: list or None
        :type k: int
        :type vote: str [majority, unanimity]
        """

        Set.__init__(self, features, labels, features_names)
        self.vote = vote
        self.k = k

    def _classify_majority(self, class_count: np.array):
        """
        Allow to classify a test dataset with a majority vote

        :param class_count: count of each class in the k neighbors

        :type class_count: np.array

        :return: np.array with the more present class and the number of presence
        """
        order_index = class_count.argsort()[::-1]
        select_index = np.where(class_count >= self.k / 2)[0]  # return only the class present in more than k/2
        # neighbors
        classify_index = [select for select in order_index if select in select_index]  # index of selected classes
        if not classify_index:  # if no classes are selected
            return np.array([None])
        else:
            return np.array([self.targets[classify_index], class_count[order_index[:len(classify_index)]]])

    def _classify_unanimity(self, class_count: np.array):
        """
        Allow to classify a test dataset with an unanimity vote

        :param class_count: count of each class in the k neighbors

        :type class_count: np.array

        :return: np.array with the more present class and the number of presence
        """
        select_index = np.where(class_count == self.k)[0]  # return only the class present in more than k/2 neighbors
        if select_index.size > 0: # if classes are selected
            return np.array([self.targets[select_index]])
        else:
            return np.array([None])

    def _classify(self, inX):
        """
        Allow to classify a test dataset

        :param inX: test feature

        :return: classification of features
        """
        eu_dist = np.sqrt(((self.features - inX) ** 2).sum(axis=1))  # Euclidian distance between train and test
        # features
        sorted_dist_indices = eu_dist.argsort()[:self.k]  # Distances sorted to keep only the K nearest

        class_count = np.zeros(self.targets_count)
        for i in sorted_dist_indices:
            # TODO parralléliser
            vote_label = self.labels[i]
            class_count[np.where(self.targets == vote_label)] += 1
        # print(class_count)
        if self.vote == 'majority':
            return self._classify_majority(class_count)
        elif self.vote == 'unanimity':
            return self._classify_unanimity(class_count)
        else:
            print("Error : vote expected method is unanimity or majority")
            return 0

    def _predict(self, classify, n_preds=1):
        if classify[0] is not None:
            if classify[0].shape[0] != n_preds:
                pred = np.empty((2, n_preds))
                if classify[0].shape[0] > n_preds:
                    pred[0] = np.stack([None] * n_preds)
                    pred[-1] = np.stack([0] * n_preds)
                else:
                    pred[0] = np.hstack([classify[0], np.stack([None] * (n_preds - len(classify[0])))])
                    pred[-1] = np.hstack([classify[-1], np.stack([0] * (n_preds - len(classify[-1])))])

            else:
                pred = classify
        else:
            pred = np.array([[None] * n_preds, [0] * n_preds])
        return pred

    def predict(self, test_set: Set, n_predict=3, vote='majority'):
        predictions = []
        # Loop through all samples, predict the class labels and store the results
        for sample in test_set:
            # TODO parralléliser
            predictions.append(self._predict(classify=self._classify(sample), n_preds=n_predict))
        self.preds = np.stack(predictions)
        return np.stack(predictions)

    def _train(self, new_train):
        return KNN(new_train[:, :-1], new_train[:, -1], k=self.k)


if __name__ == '__main__':
    import os
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    def read_file(filepath, delimiter=' '):
        contents = []
        with open(filepath) as f:
            for line in f:
                contents.append(line.strip().split(delimiter))

        return np.array(contents).astype(np.float64)


    # configuring paths
    data_dir = os.path.abspath("Data")
    print(data_dir)
    train_data = read_file(os.path.join(data_dir, "data_tp3_app.txt"))
    test_data = read_file(os.path.join(data_dir, "data_tp3_dec.txt"))
    test_KNN = Set(test_data[:, 1:], test_data[:, 0])

    # Vote Majority
    train_KNN_maj = KNN(train_data[:, 1:], train_data[:, 0], k=4, vote='majority')
    # train_KNN_maj.plot_set(title='X en fonction de Y', fig_size=[14, 9])
    train_KNN_maj.predict(test_set=test_KNN.features, n_predict=2)
    acc, rej = train_KNN_maj.accuracy(test_KNN.labels, reject=True)  # vote à l'unanimité pas de rejet
    print('Our KNN model Accuracy with majority vote : {0} with a reject rate {1}'.format(acc, rej))
    print(train_KNN_maj.confusion_matrix(test_KNN.labels, p_sum=False, reject=True))
    train_KNN_maj.plot_good_pred(test_KNN, title="Good pred", fig_size=[14, 9], reject=True)

    plot_best_k(train_KNN_maj, 2, 7, verbosity=True)

    clf = KNeighborsClassifier(n_neighbors=4)
    clf.fit(train_data[:, 1:], train_data[:, 0])
    p = clf.predict(test_data[:, 1:])
    print('Sklearn KNN model Accuracy: {}'.format(accuracy_score(test_data[:, 0], p)))
    #
    # Vote Unanimity
    train_KNN_un = KNN(train_data[:, 1:], train_data[:, 0], k=3, vote='unanimity')
    train_KNN_un.plot_set(title='X en fonction de Y', fig_size=[14, 9])
    train_KNN_un.predict(test_set=test_KNN.features, n_predict=2)
    acc, rej = train_KNN_un.accuracy(test_KNN.labels, reject=True)  # vote à l'unanimité pas de rejet
    print('Our KNN model Accuracy with unanimity vote : {0} with a reject rate {1}'.format(acc, rej))
    print(train_KNN_un.confusion_matrix(test_KNN.labels, p_sum=True, reject=True))
    train_KNN_un.plot_good_pred(test_KNN, title="Good pred", fig_size=[14, 9], reject=True)

    # Cross validation
    print(train_KNN_maj.cross_validation(n_predict=2, reject=True))
    plot_best_k(train_KNN_maj, 2, 20, verbosity=True, fig_size=[14, 9], title="Accuracy with cross-validation")
