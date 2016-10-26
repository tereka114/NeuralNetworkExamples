# coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
import argparse
import chainer.training
import chainer.optimizers
import chainer.dataset.dataset_mixin
from chainer.training import extensions
from model import ImageCaptionGenerator
import skimage.io
from dataset import get_flickr8k

class ImageCaptionDataset(chainer.dataset.dataset_mixin):
    def __init__(self, datalist):
        self.datalist = datalist

    def __len__(self):
        return len(self.datalist)

    def get_example(self, i):
        data = self.datalist[i]
        image = skimage.io.imread(data.image_path)
        return image, data.sentence


class SequentialIterator(chainer.dataset.Iterator):
    def __init__(self,dataset, batch_size, repeat=True):
        pass

    def __next__(self):
        pass


class BPTTUpdater(chainer.training.StandardUpdater):
    def __init__(self, train_iter, optimizer, bprop_len, device):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device)

    def update_core(self):
        loss = 0
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        for i in range():
            batch = train_iter.__next__()
            self.converter(batch)

            optimizer.target()

        optimizer.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choice=[])
    parser.add_argument("-g", "--gpu", default=0)

    args = parser.parse_args()

    if args.dataset == "flickr8k":
        image_caption_train_list, image_caption_test_list = get_flickr8k("./dataset/flickr8k")
    else:
        raise NotImplementedError()

    train_image_caption_dataset = ImageCaptionDataset(image_caption_train_list)
    test_image_caption_dataset = ImageCaptionDataset(image_caption_test_list)

    model = ImageCaptionGenerator()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()

    optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    updater = BPTTUpdater(train_iter, optimizer, args.bproplen, args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    SequentialIterator()