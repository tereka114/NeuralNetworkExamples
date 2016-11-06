# coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
from sklearn.base import BaseEstimator
import chainer.training
import chainer
import numpy as np


class ChainerNeuralNetworkChain(chainer.link.Chain):
    def predict(self, X, y):
        raise NotImplementedError()


class ChainerNeuralNetwork(BaseEstimator):
    def __init__(self, model, optimizer, extends=[], gpu=-1):
        """


        :param model:
        :param optimizer:
        :param extends:
        :param gpu:
        :return:
        """
        self.model = model
        self.optimizer = optimizer
        self.extends = extends
        self.gpu = gpu
        if gpu >= 0:
            chainer.cuda.get_device(gpu).use()
            self.model.to_gpu()
        super(ChainerNeuralNetwork, self).__init__()

    def convert_list_to_tuple(self, X, y):
        """
        convert list to tuple

        :param X: X
        :param y: y
        :return: tuple dataset
        """
        return chainer.datasets.TupleDataset([tuple(x, y) for x, y in zip(X, y)])

    def fit(self, X, y):
        train_iter = self.convert_list_to_tuple(X,y)
        updater = chainer.training.StandardUpdater(train_iter, self.optimizer, device=self.gpu)
        trainer = chainer.training.Trainer(updater=updater)
        for extend in self.extends:
            trainer.extend(extend)
        trainer.run()

    def _iter_predict(self, X):
        """
        iteration for prediction

        :param X: np.ndarray
        :return: prediction of batch
        """
        for index in range(len(X),self.batch_size):
            batch_x = X[index: index + self.batch_size]
            self.model.predict(batch_x).data

    def predict(self, X, y):
        return self.predict_proba()

    def predict_proba(self, X):
        return self._iter_predict(X)


