# coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
import argparse
import chainer.training
import chainer.optimizer
import chainer.optimizers
import chainer.dataset.dataset_mixin
import chainer.cuda
from chainer.training import extensions
from model import ImageCaptionGenerator, ImageFeatureModel
import skimage.io
from dataset import get_flickr8k


class ImageCaptionDataset(chainer.dataset.dataset_mixin):
    """
    Image Caption Dataset
    """

    def __init__(self, datalist):
        """
        initialize dataset

        :param datalist: list of data
        :return: None
        """
        self.datalist = datalist

    def __len__(self):
        """
        length

        :return: length
        """
        return len(self.datalist)

    def get_example(self, i):
        data = self.datalist[i]
        image = skimage.io.imread(data.image_path)
        return image, data.sentence


class SequentialIterator(chainer.dataset.Iterator):
    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.index = 0

    def get_train_data(self, dataset):
        """
        get training data

        :param dataset: list of ids
        :return:
        """
        return [dataset[index] for index in range(len(dataset) - 1)]

    def get_target_data(self, dataset):
        """
        get target data

        :param dataset: list of ids
        :return:
        """
        return [dataset[index] for index in range(1, len(dataset))]

    def __next__(self):
        """
        next iteration

        :return:
        """
        image, data = self.dataset.get_example(self.index)
        train_data = self.get_train_data(data)
        target_data = self.get_target_data(data)

        return zip(train_data, target_data)


class BPTTUpdater(chainer.training.StandardUpdater):
    def __init__(self, train_iter, optimizer, device):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device)

    def update_core(self):
        loss = 0
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        for i in range():
            img, batch = train_iter.__next__()
            x, t = self.converter(batch, self.device)
            optimizer.target.initial()

            optimizer.target()

        optimizer.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()


def convert_image_to_feature(model, images):
    """
    convert image to image feature

    :param model: extraction of image model
    :param images: list of ImageCaptionData class
    :return: None
    """
    for image in images:
        image.image_feature = model.extract(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choice=[])
    parser.add_argument("-g", "--gpu", default=0)
    parser.add_argument("-f", "--feature_model", default="./models/VGG_ILSVRC_19_layers.caffemodel")

    args = parser.parse_args()

    print("loading dataset:{}".format(args.dataset))

    if args.dataset == "flickr8k":
        image_caption_train_list, image_caption_test_list = get_flickr8k("./dataset/flickr8k")
    else:
        raise NotImplementedError()

    image_model = ImageFeatureModel(args.feature_model)

    convert_image_to_feature(image_model, image_caption_train_list)
    convert_image_to_feature(image_model, image_caption_test_list)

    interval = 10

    train_image_caption_dataset = ImageCaptionDataset(image_caption_train_list)
    test_image_caption_dataset = ImageCaptionDataset(image_caption_test_list)

    model = ImageCaptionGenerator()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()

    optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    updater = BPTTUpdater(train_image_caption_dataset, optimizer, args.bproplen, args.gpu)
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'perplexity', 'val_perplexity']
    ), trigger=(interval, 'iteration'))
    trainer.run()
