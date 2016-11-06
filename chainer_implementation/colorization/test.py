# coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
import argparse
from skimage import color
import skimage.io
from model import ColorfulImageColorizationModel

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_image")
parser.add_argument("-w", "--weight_file")

args = parser.parse_args()
input_path = args.input_image

img_rgb = skimage.io.imread(input_path)
img_lab = color.rgb2lab(img_rgb)
img_l = img_lab[:,:, 0]

# create grayscale version of image (just for displaying)

model = ColorfulImageColorizationModel()
pred = model([img_l])