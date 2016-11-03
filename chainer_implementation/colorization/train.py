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
import skimage.transform
from skimage.color import rgb2lab
import glob
from model import ColorfulImageColorizationModel
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", type=int, default=-1)
parser.add_argument("-d", "--directory", type=str, default="./dataset/VOCdevkit/VOC2012/JPEGImages")

args = parser.parse_args()


class DatasetMixin(chainer.dataset.DatasetMixin):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def get_example(self, i):
        filepath = self.X[i]
        img = skimage.io.imread(filepath)

        resize_img = skimage.transform.resize(img, (224, 224))
        lab_img = rgb2lab(resize_img).transpose((2, 0, 1))
        l_img = lab_img[0, :, :].astype(np.float32) - 50
        ab_img = lab_img[1:3, :, :].astype(np.float32)

        print l_img.shape, ab_img.shape

        return np.expand_dims(l_img, axis=0), ab_img


model = ColorfulImageColorizationModel()
if args.gpu >= 0:
    chainer.cuda.to_gpu(model)

optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

jpeg_files = glob.glob(args.directory + "/*.jpg")
dataset = DatasetMixin(jpeg_files)

train_iter = chainer.iterators.SerialIterator(dataset=dataset, batch_size=8)

updater = chainer.training.StandardUpdater(train_iter, optimizer, device=args.gpu)
trainer = chainer.training.Trainer(updater=updater)
trainer.extend(chainer.training.extensions.dump_graph('main/loss'))

trainer.extend(chainer.training.extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy'])
)
trainer.extend(chainer.training.extensions.ProgressBar())
trainer.run()
