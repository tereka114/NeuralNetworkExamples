# coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
import sklearn.datasets
from .chainer_neural_network import ChainerNeuralNetwork, ChainerNeuralNetworkChain
import chainer.links as L
import chainer.functions as F


class Classification(ChainerNeuralNetworkChain):
    def __init__(self):
        super(Classification, self).__init__(
            l1=L.Linear(None, 100),
            l2=L.Linear(100, 100),
            l3=L.Linear(100, 10)
        )

    def pred_row(self, x):
        h = self.l1(x)
        h = self.l2(h)
        h = self.l3(h)
        return h

    def __call__(self, x, y):
        h = self.pred_row()
        return F.softmax_cross_entropy(h)

    def predict(self, x, y):
        return self.pred_row(x)

sklearn.datasets.load_digits()
nn = ChainerNeuralNetwork()
