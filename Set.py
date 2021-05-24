#!/usr/bin/env python
# -*-coding:utf8-*-
from __future__ import annotations

import sys

import numpy as np
from copy import copy
import pandas as pd
import matplotlib.pyplot as plt
import itertools


def euclidian_distance(x: np.arrays, y: np.arrays):
    """
    Return the minimum euclidian distance
    Parameters:
        :param x: Points that you would know the distance from reference (y)
        :param y: Reference point
    Types:
        :type x: np.array
        :type y: np.array
    Returns
        :return:
    """
    diff = x - np.mean(y, axis=0)
    return np.sqrt(np.dot(diff.T, diff))


def mahalanobis_distance(x, y):
    """
        Return the minimum mahalanobis distance
        Parameters:
            :param x: Points that you would know the distance from reference (y)
            :param y: Reference point
        Types:
            :type x: np.array
            :type y: np.array
        Returns
            :return:
        """
    diff = x - np.mean(y, axis=0)
    cov = np.cov(y.T)  # Covariance matrix
    inv_cov = np.linalg.inv(cov)
    return np.dot(np.dot(diff, inv_cov), diff.T) + np.log(np.linalg.det(cov))


def _accuracy_without_reject(actual, preds):
    """
    Return the accuracy of a classifier without reject

    :param actual: Actuel labels for samples
    :param preds:  Predicted labels for samples

    :type actual: np.array
    :type preds: np.array

    :return: Array of n top accuracy
    """

    total = len(actual)
    acc_list = []
    for i in range(preds.shape[1]):  # for loop on the n predicted result of samples
        misclassified = sum(
            1 for act, pred in zip(actual, preds[:, i]) if act != pred)  # Number of misclassified samples
        acc_list.append((total - misclassified) / total)
    return np.cumsum(acc_list)


def _accuracy_with_reject(actual, preds):
    """
    Return the accuracy of a classifier with reject

    :param actual: Actuel labels for samples
    :param preds:  Predicted labels for samples

    :type actual: np.array
    :type preds: np.array

    :return: Array of n top accuracy
    """
    total = len(actual)
    acc_list = []
    rej_list = []
    cls_idx = 0
    while cls_idx < preds.shape[2] - 1:  # loop from nearest neighboor to furthest
        misclassified = 0
        reject = 0
        nan = 0
        for spl_idx in range(preds.shape[0]):  # loop on each sample in features
            if preds[spl_idx][0][cls_idx] != actual[spl_idx]:  # prediction and actual of the sample is different
                if preds[spl_idx][1][cls_idx] != preds[spl_idx][1][cls_idx + 1]:  # Number of prediction in top n and
                    # n+1 are different
                    misclassified += 1
                else:
                    if preds[spl_idx][1][cls_idx] != 0:  # Number of prediction is different of 0
                        reject += 1
                    else:  # Impossible to know the class
                        nan += 1
            else:
                if preds[spl_idx][1][cls_idx] == preds[spl_idx][1][cls_idx + 1] and preds[spl_idx][1][cls_idx] != 0:
                    reject += 1
                # else:
                #     nan += 1
        acc_list.append((total - misclassified - reject - nan) / total)
        rej_list.append(reject / total)
        cls_idx += 1
    goodclassified = sum(1 for act, pred in zip(actual, preds[:, 0, cls_idx]) if act == pred)
    acc_list.append(goodclassified / total)
    rej_list.append(0)
    return np.cumsum(acc_list), rej_list


def plot_accuracy(parameters, accuracies, rejects=None, fig_size=None, title=None):
    if fig_size is not None:
        fig = plt.figure(figsize=fig_size)
    else:
        fig = plt.figure()
    ax = fig.add_subplot(111)
    for idx, acc in enumerate(accuracies):
        plt.plot(parameters, acc, label="accuracy top{0}".format(idx + 1))
    if rejects is not None:
        idx = 0
        while idx < len(rejects)-1:
            plt.plot(parameters, rejects[idx, :], label="reject top{0}".format(idx + 1))
            idx += 1
        plt.ylabel('accuracy & reject')
    else:
        plt.ylabel('accuracy')
    plt.xlabel('parameters')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if title is not None:
        plt.title(title)
    plt.show()


class Set:
    """
        A class to create a set of data for machine learning analyses

        Attributes
        ----------
        features : np.array
            Array corresponding to all the variables in the dataset
        labels : np.array
            Array corresponding to all the targets in the dataset
        features_names : list or None
            list corresponding to the name of each variable.

        Methods
        -------
        train_test_split(self, test_size=0.1, random_state=0, random_rate_label=True):
            split a Set in a train set and test set or in train variable arrays, train target array, test variable array
            and test array.
        """

    def __init__(self, features: np.array, labels: np.array, features_names=None):
        """
        Parameters
        ----------
        features : np.array
            Array corresponding to all the variables in the dataset
        labels : np.array
            Array corresponding to all the targets in the dataset
        features_names : list or None
            list corresponding to the name of each variable.
        """

        self.features = np.array(features)
        self.features_names = features_names
        self.labels = np.array(labels)
        self.targets, self.labels_counts = np.unique(self.labels, return_counts=True)  # Target corresponding to all
        # the different labels, labels_count to the frequency of each classe
        self.targets_count = self.targets.shape[0]  # Number of different classe
        self.preds = None  # Contain all the prediction compute with a test Set

    def __repr__(self):
        df = pd.DataFrame(np.column_stack((self.features, self.labels)))
        df.set_axis([*df.columns[:-1], 'Classe'], axis=1, inplace=False)
        return repr(df)

    def _count_targets(self, array: np.array):
        idx = np.searchsorted(self.targets, array)
        idx[idx == len(self.targets)] = 0
        mask = self.targets[idx] == array
        return np.bincount(idx[mask], minlength=self.targets_count)

    def _predict(self, classify: np.array, n_preds=1):
        """
        Return the n best target predicted in function of the classification results

        :param classify: Classification of the different labels for samples
        :param n_preds: Number of labels return

        :type classify: np.array
        :type n_preds: int

        :return: n best labels for samples
        """
        tmp = classify.argsort()[:, :n_preds]  # Return the index of the best label classification
        preds = copy(tmp)  # allow to copy tmp
        for index, target in enumerate(self.targets):
            preds = np.where(tmp == index, target, preds)  # Return the target label corresponding to the index
        self.preds = preds

    def _confusion_matrix(self, actual, classify, p_sum, reject):
        """
        Return a confusion matrix between the prediction and the actual labels

        :param actual: Real labels from a test Set
        :param classify: Classification results if the prediction is not done yet
        :param p_sum: Sum the column and line of the matrix
        :param reject: True if the classifier return reject in prediction

        :type actual: np.array
        :type classify: np.array
        :type p_sum : Boolean
        :type reject : Boolean

        :return: pandas dataframe corresponding to the confusion matrix
        """
        if self.preds is None:
            self._predict(classify)
        x_actu = pd.Series(actual, name='Actual')
        if reject:
            y_pred = pd.Series(self.preds[:, 0, 0], name='Predicted')
        else:
            y_pred = pd.Series(self.preds[:, 0], name='Predicted')
        if len(pd.Series(pd.unique(y_pred)).dropna()) == len(np.unique(actual)):  # Check if the number of different
            # target in y pred is the same than is actual
            return pd.crosstab(x_actu, y_pred, margins=p_sum, dropna=False)
        else:
            df = pd.crosstab(x_actu, y_pred, margins=p_sum, dropna=False)
            mask = np.in1d(np.unique(actual), np.unique(y_pred))  # Add the missing targets to y_pred
            if p_sum:
                column_z = [0] * (len(np.unique(actual)) + 1)
            else:
                column_z = [0] * len(np.unique(actual))
            for idx in np.where(~mask)[0]:
                df.insert(loc=int(idx), column=self.targets[idx], value=column_z)  # Add a zero column in the matrix
            return df

    def _normalized_confusion_matrix(self, actual, classify, p_sum, reject):
        """
        Return a normalized confusion matrix between the prediction and the actual labels

        :param actual: Real labels from a test Set
        :param classify: Classification results if the prediction is not done yet
        :param p_sum: Sum the column and line of the matrix
        :param reject: True if the classifier return reject in prediction

        :type actual: np.array
        :type classify: np.array
        :type p_sum : Boolean
        :type reject : Boolean

        :return: pandas dataframe corresponding to the normalized confusion matrix
        """
        df_confusion = self._confusion_matrix(actual, classify, p_sum, reject)
        return df_confusion / df_confusion.sum(axis=1)

    def confusion_matrix(self, actual, classify=None, p_sum=False, norm=False, reject=False):
        """
        Return a confusion matrix between the prediction and the actual labels

        :param actual: Real labels from a test Set
        :param classify: Classification results if the prediction is not done yet
        :param p_sum: Sum the column and line of the matrix
        :param norm: return the normalised version of the matrix
        :param reject: True if the classifier return reject in prediction

        :type actual: np.array
        :type classify: np.array
        :type p_sum : Boolean
        :type norm : Boolean
        :type reject : Boolean

        :return: pandas dataframe corresponding to the confusion matrix
        """
        if norm:
            if p_sum:
                print("ERROR : Sum normalized confusion is not avail yet")
                exit()
            else:
                return self._normalized_confusion_matrix(actual, classify, p_sum, reject)
        else:
            return self._confusion_matrix(actual, classify, p_sum, reject)

    def _shuffle_set(self, random_state, replace=False):
        np.random.seed(random_state)
        dataset = np.column_stack((self.features, self.labels)).copy()
        if replace:
            return dataset[np.random.choice(dataset.shape[0], dataset.shape[0], replace=replace), :]
        else:
            np.random.shuffle(dataset)
            return dataset

    def _split_set(self, n_fold=5, random_state=0):
        data = self._shuffle_set(random_state)
        return np.array_split(data, n_fold)

    def _split_set_index(self, size: int):
        select_index = []
        for target in self.targets:
            index_target = np.where(self.labels == target)[0]
            select_index.append(np.random.choice(index_target, size, replace=False))
        return np.hstack(select_index)

    def plot_set(self, title=None, fig_size=None, close=True):
        """
        Return a plot of the Set variable with a color for each target

        :param title: Set the plot's title
        :param fig_size: Set the figure size
        :param close: True to close and show the figure

        :type title: str
        :type fig_size: tupple
        :type close: Boolean

        :return: plot
        """
        if self.features.shape[1] == 2:
            return self.plot_2D(title, fig_size, close)
        elif self.features.shape[1] >= 3:
            return self.plot_3D(title, fig_size, close)
        else:
            print("We can't generate a graph with {0} dimensions".format(self.features.shape[1]))
            sys.exit()

    def plot_2D(self, title=None, fig_size=None, close=True):
        # TODO add possibility to change title parameter
        """
        Return a plot of the Set variable with a color for each target

        :param title: Set the plot's title
        :param fig_size: Set the figure size
        :param close: True to close and show the figure

        :type title: str
        :type fig_size: tupple
        :type close: Boolean

        :return: 2D plot
        """

        if fig_size is not None:
            fig = plt.figure(figsize=fig_size)
        else:
            fig = plt.figure()
        ax = fig.add_subplot(111)
        for target in self.targets:
            idx = np.where(self.labels == target)
            ax.scatter(self.features[idx, 0], self.features[idx, 1], label=str(target))
        if self.features_names is not None:
            plt.xlabel(str(self.features_names[0]))
            plt.ylabel(str(self.features_names[1]))
        else:
            plt.xlabel('axe 1')
            plt.ylabel('axe 2')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if title is not None:
            plt.title(title)
        if close:
            plt.show()
        else:
            return fig

    def plot_3D(self, title=None, fig_size=None, close=True):
        # TODO ajouter des titres
        """
        Return a multiple plot of the Set variable with a color for each target

        :param title: Set the plot's title
        :param fig_size: Set the figure size
        :param close: True to close and show the figure

        :type title: str
        :param fig_size: tupple
        :param close: Boolean

        :return: multiple 3D plot
        """
        combs = list(itertools.combinations(np.arange(self.features.shape[1]), 3))
        idx_plot = 1
        if fig_size is not None:
            fig = plt.figure(figsize=fig_size)
        else:
            fig = plt.figure()
        if len(combs) % 2 == 1:
            n_col, n_row = (int((len(combs) + 1) / 2), int(len(combs) / 2))
        else:
            n_col, n_row = (int(len(combs) / 2), int(len(combs) / 2))
        for x, y, z in combs:
            ax = fig.add_subplot(n_row, n_col, idx_plot, projection='3d')
            for target in self.targets:
                idx = np.where(self.labels == target)
                ax.scatter(self.features[idx, x], self.features[idx, y], self.features[idx, z], label=str(target))
            if self.features_names is not None:
                ax.set_xlabel(str(self.features_names[x]))
                ax.set_ylabel(str(self.features_names[y]))
                ax.set_zlabel(str(self.features_names[z]))
            if title is not None:
                ax.set_title(title[idx_plot - 1])
                idx_plot += 1
            plt.legend(fontsize='small')
        if close:
            plt.show()
        else:
            return fig

    def _plot_good_pred_whitout_reject(self, test: Set, title=None, fig_size=None):
        """
        Return a plot of the prediction with a color for good prediction and misclassified

        :param test: data test Set
        :param title: Set the plot's title
        :param fig_size: Set the figure size

        :type test: Set
        :type title: str
        :param fig_size: tupple

        :return: 2D plot
        """
        if fig_size is not None:
            fig = plt.figure(figsize=fig_size)
        else:
            fig = plt.figure()
        ax = fig.add_subplot(111)
        goodclassified_index = []
        for idx_preds in range(self.preds.shape[1]):
            new_good_index = []
            for idx in range(self.preds.shape[0]):
                if test.labels[idx] == self.preds[idx, idx_preds]:
                    new_good_index.append(idx)
            if new_good_index:
                ax.scatter(test.features[new_good_index, 0], self.features[new_good_index, 1],
                           label='Good classified top{0}'.format(int(idx_preds + 1)))
            goodclassified_index += new_good_index
        misclassified = [idx for idx in range(self.preds.shape[0]) if idx not in goodclassified_index]
        if misclassified:
            ax.scatter(test.features[misclassified, 0], test.features[misclassified, 1],
                       label='Misclassified', marker='x', c='red')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if title is not None:
            ax.set_title(title)
        plt.show()

    def _plot_good_pred_whit_reject(self, test: Set, title=None, fig_size=None):
        """
        Return a plot of the prediction with a color for good prediction, misclassified and reject

        :param test: data test Set
        :param title: Set the plot's title
        :param fig_size: Set the figure size

        :type test: Set
        :type title: str
        :param fig_size: tupple

        :return: 2D plot
        """
        if fig_size is not None:
            fig = plt.figure(figsize=fig_size)
        else:
            fig = plt.figure()
        ax = fig.add_subplot(111)
        goodclassified_index = []
        for idx_preds in range(self.preds.shape[1] - 1):
            new_good_index = []
            for idx in range(self.preds.shape[0]):
                if self.preds[idx][0][idx_preds] == test.labels[idx] and \
                        self.preds[idx][1][idx_preds] != self.preds[idx][1][idx_preds + 1]:
                    new_good_index.append(idx)
            if new_good_index:
                ax.scatter(test.features[new_good_index, 0], self.features[new_good_index, 1],
                           label='Good classified top{0}'.format(int(idx_preds + 1)))
            goodclassified_index += new_good_index
        new_good_index = []
        for idx in range(self.preds.shape[0]):
            if self.preds[idx][0][-1] == test.labels[idx]:
                new_good_index.append(idx)
        if new_good_index:
            ax.scatter(test.features[new_good_index, 0], self.features[new_good_index, 1],
                       label='Good classified top{0}'.format(int(self.preds.shape[1])))
        goodclassified_index += new_good_index
        reject_idx, misclassified_idx = ([], [])
        for idx in range(self.preds.shape[0]):
            if idx not in goodclassified_index:
                reject = False
                for idx_preds in range(self.preds.shape[1] - 1):
                    if self.preds[idx][1][idx_preds] == self.preds[idx][1][idx_preds + 1]:
                        reject_idx.append(idx)
                        reject = True
                        break
                if not reject:
                    misclassified_idx.append(idx)
        if reject_idx:
            ax.scatter(test.features[reject_idx, 0], self.features[reject_idx, 1],
                       label='Reject', c='orange', marker='^')
        if misclassified_idx:
            ax.scatter(test.features[misclassified_idx, 0], self.features[misclassified_idx, 1],
                       label='Misclassified', marker='x', c='red')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if title is not None:
            ax.set_title(title)
        plt.show()

    def plot_good_pred(self, test: Set, title=None, fig_size=None, reject=False):
        """
        Return a plot of the prediction with a color for good prediction, misclassified and reject

        :param test: data test Set
        :param title: Set the plot's title
        :param fig_size: Set the figure size
        :param reject: True to close and show the figure

        :type test: Set
        :type title: str
        :type fig_size: tupple
        :type reject: Boolean

        :return: 2D plot
        """
        if reject:
            self._plot_good_pred_whit_reject(test, title, fig_size)
        else:
            self._plot_good_pred_whitout_reject(test, title, fig_size)

    def train_test_split(self, test_size=0.1, random_state=0, random_rate_label=True):
        """
        Split a Set in a train set and test set or in train variable arrays, train target array, test variable array
            and test array.
        Parameters:
            :param test_size:
            :param random_state:
            :param random_rate_label:
        Types:
            :type test_size:
            :type random_state:
            :type random_rate_label:
        Returns:
            :return:
        """
        data = self._shuffle_set(random_state)
        if test_size < 1:
            threshold = int(data.shape[0] * test_size)
        else:
            threshold = test_size
        dataset = Set(data[:, :-1], data[:, -1])
        if random_rate_label:
            split_index = np.random.randint(dataset.labels.shape[0], size=threshold)
        else:
            split_index = dataset._split_set_index(size=int(threshold / self.targets_count))
        X_test = dataset.features[split_index, :]
        y_test = dataset.labels[split_index]
        X_train = np.delete(dataset.features, split_index, axis=0)
        y_train = np.delete(dataset.labels, split_index, axis=0)
        return X_train, y_train, X_test, y_test

    def _train(self, new_train):
        return Set(new_train[:, :-1], new_train[:, -1])

    def cross_validation(self, n_fold=5, random_state=0, n_predict=2, reject=False, conf_matrix=False):
        """

        :param n_fold: number of folds
        :param random_state: makes the random numbers predictable
        :param n_predict: Number of labels return
        :param reject: True for classifier with reject
        :param conf_matrix: True to obtain the confusion matrix list

        :type n_fold: int
        :type random_state: int
        :type n_predict: int
        :type reject: Boolean
        :type conf_matrix: Boolean

        :return: Mean accuracy of the classifier and a list of confusion matrix
        """
        split_list = self._split_set(n_fold, random_state)
        acc_list = []
        if reject:
            rej_list = []
        conf_mat_list = []
        for fold in range(n_fold):
            test_set = Set(split_list[fold][:, :-1], split_list[fold][:, -1])
            new_train = np.vstack([Xy for i, Xy in enumerate(split_list) if i != fold])  # Create a train dataset
            train_set = self.__class__._train(self, new_train)  # Call the class of self object to generate the train
            train_set.predict(test_set.features, n_predict)
            if reject:
                acc, rej = train_set.accuracy(test_set.labels, reject=reject)
                acc_list.append(acc)
                rej_list.append(rej)
                if conf_matrix:
                    conf_mat_list.append(train_set.confusion_matrix(test_set.labels, reject=True))
            else:
                acc_list.append(train_set.accuracy(test_set.labels, reject=reject))
                if conf_matrix:
                    conf_mat_list.append(train_set.confusion_matrix(test_set.labels))
        if reject:
            if conf_matrix:
                return np.mean(acc_list, axis=0), np.mean(rej_list, axis=0), conf_mat_list
            else:
                return np.mean(acc_list, axis=0), np.mean(rej_list, axis=0)
        else:
            if conf_matrix:
                return np.mean(acc_list, axis=0), conf_mat_list
            else:
                print(acc_list)
                return np.mean(acc_list, axis=0)

    def bagging(self, test_set: Set, n=5, random_state=None, vote='majority', reject=False,
                accuracy=False, conf_matrix=False):
        i = 0
        preds_list = []
        while i < n:
            i += 1
            new_train_data = self.__class__._train(self, new_train=self._shuffle_set(random_state, replace=True))
            new_train_data.predict(test_set.features, n_predict=1)
            if reject:
                # print(new_train_data.preds)
                preds_list.append(new_train_data.preds[:, 0, 0])
        bagging = np.stack(preds_list, axis=0)
        bagging_count = np.stack([np.count_nonzero(bagging == target, axis=0) for target in self.targets], axis=0)
        if vote == 'majority':
            index = np.where(bagging_count >= n / 2, 1, 0)
            self.preds = bagging[np.where(index >= 1)]
            self.preds = np.insert(self.preds, np.where(np.sum(index, axis=0) == 0)[0], None)
            self.preds = np.array([[elem] for elem in self.preds])
        else:
            print("No other vote is avail yet")
            sys.exit()
        if accuracy:
            acc = self.accuracy(test_set.labels, reject=False)
            if conf_matrix:
                return acc, self.confusion_matrix(test_set.labels, reject=False)
            else:
                return acc
        else:
            if conf_matrix:
                return self.confusion_matrix(test_set.labels, reject=False)

    def accuracy(self, actual, reject=False):
        """
        Return the accuracy of a classifier with the difference between the prediction and the actual labels

        :param actual: Real labels from a test Set
        :param reject: True for classifier with reject

        :type actual: np.array
        :type reject: Boolean

        :return: array of good prediction rate and reject rate if reject True
        """
        if not reject:
            return _accuracy_without_reject(preds=self.preds,
                                            actual=actual)  # Return accuracy for classifier without reject
        else:
            return _accuracy_with_reject(preds=self.preds, actual=actual)  # Return accuracy for classifier with reject

    def predict(self, features, n_predict):
        print("No predict methods for a set object")
        sys.exit()


class Euclidean(Set):
    def __init__(self, features: np.array, labels: np.array, features_names=None):
        Set.__init__(self, features, labels, features_names)

    def _euclidian_classifier(self, X_test: np.array, y_test: np.array):
        """
        Return the classification of all test samples
        :param X_test: Features from the test dataset
        :param y_test: labels from the test dataset

        :type X_test: np.array
        :type y_test: np.array

        :return: Euclidian distance between test features and train features for all targets
        """
        dist = np.empty([X_test.shape[0], y_test.shape[0]])
        for index, target in enumerate(self.targets):
            dist[:, index] = np.array([euclidian_distance(sample, self.features[np.where(self.labels == target)])
                                       for sample in X_test])
        return dist

    def _classifier(self, test_set):
        """
        Return the classification of all test samples
        :param test_set: test dataset

        :type test_set: Set

        :return: Euclidian distance between test features and train features for all targets
        """
        return self._euclidian_classifier(test_set.features, test_set.targets)

    def predict(self, test_set=None, n_preds=1):
        if test_set is None:
            print("test_set is required")
            sys.exit()
        self._predict(self._classifier(test_set), n_preds)


class Mahalanobis(Set):
    def __init__(self, features: np.array, labels: np.array, features_names=None):
        Set.__init__(self, features, labels, features_names)

    def _mahalanobis_classifier(self, X_test: np.array, y_test: np.array):
        """
        Return the euclidian distance between test features and train features for all targets
        :param X_test: Features from the test dataset
        :param y_test: labels from the test dataset

        :type X_test: np.array
        :type y_test: np.array

        :return: Mahalanobis distance between test features and train features for all targets
        """

        dist = np.empty([X_test.shape[0], y_test.shape[0]])
        for index, target in enumerate(self.targets):
            dist[:, index] = np.array([mahalanobis_distance(sample, self.features[np.where(self.labels == target)])
                                       for sample in X_test])
        return dist

    def _classifier(self, test_set):
        """
        Return the classification of all test samples
        :param test_set: test dataset

        :type test_set: Set

        :return: Mahalanobis distance between test features and train features for all targets
        """
        return self._mahalanobis_classifier(test_set.features, self.targets)

    def predict(self, test_set=None, n_preds=1):
        if test_set is None:
            print("test_set is required")
            sys.exit()
        self._predict(self._classifier(test_set), n_preds)


if __name__ == '__main__':
    import os


    def read_file(filepath, delimiter=' '):
        contents = []
        with open(filepath) as f:
            for line in f:
                contents.append(line.strip().split(delimiter))

        return np.array(contents).astype(np.float64)


    data_dir = os.path.abspath("Data")
    train_data = read_file(os.path.join(data_dir, "data_tp1_app.txt"))
    test_data = read_file(os.path.join(data_dir, "data_tp1_dec.txt"))

    # Euclidean minimum distance classify
    train_eu = Euclidean(train_data[:, 1:], train_data[:, 0])
    test_eu = Euclidean(test_data[:, 1:], test_data[:, 0])
    train_eu.predict(n_preds=2, test_set=test_eu)
    print('Euclidean distance minimum model Accuracy: {}'.format(train_eu.accuracy(test_eu.labels,
                                                                                   reject=False)))
    print(train_eu.confusion_matrix(test_eu.labels))
    train_eu.plot_good_pred(test_eu, title='Accuracy TP1 with euclidean distances', fig_size=[14, 9])

    # Mahalanobis minimum distance classify
    train_ma = Mahalanobis(train_data[:, 1:], train_data[:, 0])
    test_ma = Mahalanobis(test_data[:, 1:], test_data[:, 0])
    train_ma.predict(n_preds=2, test_set=test_ma)

    print('Mahalanobis distance minimum model Accuracy: {}'.format(train_ma.accuracy(test_ma.labels,
                                                                                     reject=False)))
    print(train_ma.confusion_matrix(test_ma.labels, p_sum=True))
    train_ma.plot_good_pred(test_eu, title='Accuracy TP1 with mahalanobis distances', fig_size=[14, 9])

    # Split Data
    data = Set(test_data[:, 1:], test_data[:, 0], features_names=['X', 'Y'])
    data.plot_2D(title='X en fonction de Y', fig_size=[14, 9])
    X_train, y_train, X_test, y_test = data.train_test_split()
    train = Euclidean(X_train, y_train)
    test = Euclidean(X_test, y_test)
    train.predict(n_preds=2, test_set=test)
    print('Euclidean distance minimum model Accuracy with train_test_split: {}'.format(train.accuracy(test.labels,
                                                                                                      reject=False)))
    print(train.confusion_matrix(test.labels, norm=True))

    train = Mahalanobis(X_train, y_train)
    test = Mahalanobis(X_test, y_test)
    train.predict(n_preds=2, test_set=test)
    print('Mahalanobis distance minimum model Accuracy with train_test_split: {}'.format(train.accuracy(test.labels,
                                                                                                        reject=False)))
    print(train.confusion_matrix(test.labels))
    test.plot_set(fig_size=[14, 9])
    train.plot_good_pred(test, title='Accuracy', fig_size=[14, 9])

    data_bank = read_file(os.path.join(data_dir, "data_banknote_authentication.txt"), delimiter=',')
    set_bank = Set(data_bank[:, :-1], data_bank[:, -1], features_names=['X', 'Y', 'Z', 'T'])
    set_bank.plot_3D(title=['A', 'B', 'C', 'D'], fig_size=[14, 9])

    X_train, Y_train, X_test, Y_test = set_bank.train_test_split(test_size=0.1, random_state=7)
    test_bank = Set(X_test, Y_test,  features_names=['X', 'Y', 'Z', 'T'])

    # Euclidean minimum distance classify
    train_eu_bank = Euclidean(X_train, Y_train,  features_names=['X', 'Y', 'Z', 'T'])
    train_eu_bank.predict(n_preds=2, test_set=test_bank)
    print('Euclidean distance minimum model Accuracy: {}'.format(train_eu_bank.accuracy(test_bank.labels,
                                                                                        reject=False)))
    print(train_eu_bank.confusion_matrix(test_bank.labels))
    train_eu_bank.plot_good_pred(test_bank, title='Accuracy TP1 with euclidean distances', fig_size=[14, 9])

    # Mahalanobis minimum distance classify
    train_ma_bank = Mahalanobis(X_train, Y_train, features_names=['X', 'Y', 'Z', 'T'])
    train_ma_bank.predict(n_preds=2, test_set=test_bank)
    print('Euclidean distance minimum model Accuracy: {}'.format(train_ma_bank.accuracy(test_bank.labels,
                                                                                        reject=False)))
    print(train_ma_bank.confusion_matrix(test_bank.labels))
    train_ma_bank.plot_good_pred(test_bank, title='Accuracy TP1 with euclidean distances', fig_size=[14, 9])
