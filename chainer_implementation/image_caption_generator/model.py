# coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
import chainer.links as L
import chainer.functions as F
from chainer.functions import caffe
from skimage.io import imread
from skimage.transform import resize
import chainer


class ImageFeatureModel(object):
    """
    This model is using pretrained VGG19 model
    """

    def __init__(self, path):
        """
        Extract Image Feature Function

        :param path: vgg19 filepath
        :return:None
        """
        self.function = caffe.CaffeFunction(path)

    def convert(self, image_path):
        image = imread(image_path)
        image = resize(image, (224, 224))

        image[:][:][0] -
        image[:][:][1] -
        image[:][:][2] -

        x = chainer.Variable(image)
        feature = self.function(inputs={'data': x}, outputs=['fc7'])
        return feature


class ImageCaptionGenerator(chainer.Chain):
    """
    This is image caption generator
    """

    def __init__(self, image_feature_number, word_length, hidden_num):
        super(ImageCaptionGenerator, self).__init__(
            image_vec=L.Linear(image_feature_number, hidden_num),
            word_vec=L.EmbedID(word_length, hidden_num),
            lstm=L.LSTM(hidden_num, hidden_num),
            output=L.Linear(hidden_num, word_length)
        )

    def initialize(self, image_feature):
        """
        first input

        :param image_feature: image feature
        :return:
        """
        self.lstm.reset_state()
        h = self.image_vec(image_feature)
        self.lstm(h)

    def predict(self, x):
        """
        predict x

        :param x: input
        :return: output
        """
        h = self.word_vec(x)
        h = self.lstm(h)
        h = self.word_vec(h)
        return h

    def __call__(self, x, t):
        h = self.predict(x)
        return F.softmax_cross_entropy(h, t)
