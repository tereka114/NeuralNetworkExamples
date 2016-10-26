# coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
import chainer
import chainer.links as L
import chainer.functions as F


class NTM(chainer.Chain):
    def __init__(self):
        super(NTM, self).__init__(
            controller=L.LSTM(),
            k=L.Linear(),
            g=L.Linear(),
            s_w=L.Linear(),
            beta=L.Linear(),
            gamma=L.Linear()
        )

    def build_write_head(self):
        pass

    def build_read_head(self):
        pass

    def build_head(self,x):
        k = F.tanh(self.k(x))
        g = F.sigmoid(self.g(x))
        s_w = F.softmax(self.s_w(x))
        beta = F.softplus(self.beta(x))

        # Sharpeing


    def __call__(self, x):
        """
        this is one step of ntm

        :param x:
        :return:
        """
        h = self.controller(x)
