#coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
import argparse
from model import ImageCaptionGenerator, ImageFeatureModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--feature_model", default="./models/VGG_ILSVRC_19_layers.caffemodel")
    parser.add_argument("-m", "--model", default="")
    parser.add_argument("-i", "--image_path", default="")

    args = parser.parse_args()

    image_feature_model = ImageFeatureModel()

    image_caption_generator = ImageCaptionGenerator()

    image_caption_generator.initialize()

    for token in tokens:
        pred = image_caption_generator.predict(token)