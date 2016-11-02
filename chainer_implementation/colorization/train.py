# coding:utf-8
from __future__ import absolute_import
import chainer.training
import chainer.dataset
import chainer.optimizers
import chainer.iterators
import chainer.training.extensions
import chainer.cuda

import argparse
import skimage.io
from skimage.color import rgb2lab
import glob
from model import ColorfulImageColorizationModel

parser = argparse.ArgumentParser()
parser.add_argument("-k", "")
parser.add_argument("-g", "--gpu", type=int, default=-1)
parser.add_argument("-d", "--directory", type=str, default="./dataset/VOCdevkit/VOC2012/JPEGImages")

args = parser.parse_args()


class DatasetMixin(chainer.dataset.DatasetMixin):
    def __init__(self, X):
        self.X = X

    def get_example(self, i):
        filepath = self.X[i]
        img = skimage.io.imread(filepath)
        lab_img = rgb2lab(img)
        l_img = lab_img[:, :, 0]
        ab_img = lab_img[:, :, 1:2]

        return l_img, ab_img


model = ColorfulImageColorizationModel()
if args.gpu >= 0:
    chainer.cuda.to_gpu(model)

optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

#prepare datasets
jpeg_files = glob.glob(args.directory + "/*.jpg")

train_iter = chainer.iterators.SerialIterator()
test_iter = chainer.iterators.SerialIterator()

updater = chainer.training.StandardUpdater(train_iter, optimizer, device=args.gpu)
trainer = chainer.training.Trainer(updater=updater)
trainer.extend(chainer.training.extensions.dump_graph('main/loss'))

trainer.extend(chainer.training.extensions.PrintReport(

))
trainer.extend(chainer.training.extensions.ProgressBar())
trainer.run()
