# coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
import chainer
import chainer.links as L


class VGG19(chainer.Chain):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


class ImageCaptionGenerator(chainer.Chain):
    def __init__(self):
        super(ImageCaptionGenerator, self).__init__(
            image_vec=L.Linear(None),
            word_vec=L.EmbedID(None),
            lstm=L.LSTM(),
            word_vec=L.Linear(None)
        )

    def __call__(self,image_feature):
