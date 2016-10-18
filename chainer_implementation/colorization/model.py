# coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
# coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
import chainer.links as L
import chainer.functions as F
from prior_boost import PriorBoostLayer
from nn_enc import NNEncLayer


class ColorfulImageColorizationModel():
    """
    Colorful Image Colorization Model


    """
    def __init__(self):
        conv1 = L.Convolution2D()
        norm1 = L.BatchNormalization()
        conv2 = L.Convolution2D()
        conv3 = L.Convolution2D()
        conv3 = L.Convolution2D()
        conv3 = L.Convolution2D()
        conv3 = L.Convolution2D()
        conv3 = L.Convolution2D()

        conv313 = L.Convolution2D()
        # TODO: define layer
        self.prior_boost_layer = PriorBoostLayer()
        self.nn_enc_layer = NNEncLayer()

    def __call__(self, x, y):
        # input L image and ab image
        F.relu(self.conv1(x))
