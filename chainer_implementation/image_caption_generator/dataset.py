# coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
import skimage
import os
import json
import collections


class ImageCaptionData(object):
    def __init__(self, image_path, sentence):
        self.image_path = image_path
        self.sentence = sentence


def get_flickr8k(dir=""):
    """
    get flickr8k dataset

    :param dir: directory
    :return: data list
    """
    dataset_path = os.path.join(dir, "dataset.json")
    json_data = json.load(open(dataset_path, "r"))
    print json_data["images"][0].keys()
    dictionary = {}
    id = 0

    images = json_data["images"]
    # create dictionary
    for data in images:
        tokens = data["sentences"][0]["tokens"]
        for token in tokens:
            if not dictionary.has_key(token):
                dictionary[token] = id
            else:
                pass

    image_caption_train_list = []
    image_caption_test_list = []

    for data in images:
        tokens = data["sentences"][0]["tokens"]
        id_tokens = [dictionary[token] for token in tokens]
        image_filename = data["filename"]
        image_caption_data = ImageCaptionData(image_filename, id_tokens)

        if data["split"] == "train":
            image_caption_train_list.append(image_caption_data)
        elif data["split"] == "test":
            image_caption_test_list.append(image_caption_data)

    return image_caption_train_list, image_caption_test_list, dictionary


if __name__ == '__main__':
    get_flickr8k("./dataset/flickr8k")
