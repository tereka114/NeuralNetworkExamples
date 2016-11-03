# coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
# coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
import chainer
import chainer.links as L
import chainer.functions as F
from training_layers import PriorBoostLayer, NNEncLayer, ClassRebalanceMultLayer, NonGrayMaskLayer


class ColorfulImageColorizationModel(chainer.Chain):
    """
    Colorful Image Colorization Model
    """

    def __init__(self):
        super(ColorfulImageColorizationModel, self).__init__(
            data_ab_ss=L.Convolution2D(None, 2, ksize=1, stride=4),
            conv1_1=L.Convolution2D(None, 64, ksize=3, pad=1),
            conv1_2=L.Convolution2D(None, 64, ksize=3, pad=1, stride=2),
            conv1_2norm=L.BatchNormalization(64),
            conv2_1=L.Convolution2D(None, 128, ksize=3, pad=1),
            conv2_2=L.Convolution2D(None, 128, ksize=3, pad=1, stride=2),
            conv2_2norm=L.BatchNormalization(128),
            conv3_1=L.Convolution2D(None, 256, ksize=3, pad=1),
            conv3_2=L.Convolution2D(None, 256, ksize=3, pad=1),
            conv3_3=L.Convolution2D(None, 256, ksize=3, pad=1, stride=2),
            conv3_3norm=L.BatchNormalization(256),
            conv4_1=L.DilatedConvolution2D(256, 512, ksize=3, pad=1, dilate=1),
            conv4_2=L.DilatedConvolution2D(512, 512, ksize=3, pad=1, dilate=1),
            conv4_3=L.DilatedConvolution2D(512, 512, ksize=3, pad=1, dilate=1),
            conv4_3norm=L.BatchNormalization(512),
            conv5_1=L.DilatedConvolution2D(512, 512, ksize=3, pad=2, dilate=2),
            conv5_2=L.DilatedConvolution2D(512, 512, ksize=3, pad=2, dilate=2),
            conv5_3=L.DilatedConvolution2D(512, 512, ksize=3, pad=2, dilate=2),
            conv5_3norm=L.BatchNormalization(512),
            conv6_1=L.DilatedConvolution2D(512, 512, ksize=3, pad=2, dilate=2),
            conv6_2=L.DilatedConvolution2D(512, 512, ksize=3, pad=2, dilate=2),
            conv6_3=L.DilatedConvolution2D(512, 512, ksize=3, pad=2, dilate=2),
            conv6_3norm=L.BatchNormalization(512),
            conv7_1=L.DilatedConvolution2D(512, 512, ksize=3, pad=1, dilate=1),
            conv7_2=L.DilatedConvolution2D(512, 512, ksize=3, pad=1, dilate=1),
            conv7_3=L.DilatedConvolution2D(512, 512, ksize=3, pad=1, dilate=1),
            conv7_3norm=L.BatchNormalization(512),
            conv8_1=L.Deconvolution2D(512, 256, ksize=4, pad=1, stride=2),
            conv8_2=L.DilatedConvolution2D(256, 256, ksize=3, pad=1, dilate=1),
            conv8_3=L.DilatedConvolution2D(256, 256, ksize=3, pad=1, dilate=1),
            conv313=L.DilatedConvolution2D(256, 313, ksize=1, dilate=1)
        )

        self.prior_boost_layer = PriorBoostLayer(

        )
        self.nn_enc_layer = NNEncLayer()
        # self.class_reblance_layer = ClassRebalanceMultLayer()
        self.non_gray_mask_layer = NonGrayMaskLayer()

    def __call__(self, x, y):
        data_ab_ss = self.data_ab_ss(y)
        gt_ab_313 = self.nn_enc_layer.forward(data_ab_ss.data)
        gt_ab_313_va = chainer.Variable(gt_ab_313)

        # non_gray_mask = self.non_gray_mask_layer.forward(data_ab_ss)
        # prior_boost = self.prior_boost_layer.forward(gt_ab_313)

        # prior_boost_nongray = prior_boost * non_gray_mask

        h = F.relu(self.conv1_1(x))
        h = self.conv1_2norm(F.relu(self.conv1_2(h)))

        h = F.relu(self.conv2_1(h))
        h = self.conv2_2norm(F.relu(self.conv2_2(h)))

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = self.conv3_3norm(h)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = self.conv4_3norm(h)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = self.conv5_3norm(h)

        h = F.relu(self.conv6_1(h))
        h = F.relu(self.conv6_2(h))
        h = F.relu(self.conv6_3(h))
        h = self.conv6_3norm(h)

        h = F.relu(self.conv7_1(h))
        h = F.relu(self.conv7_2(h))
        h = F.relu(self.conv7_3(h))
        h = self.conv7_3norm(h)

        h = F.relu(self.conv8_1(h))
        h = F.relu(self.conv8_2(h))
        h = F.relu(self.conv8_3(h))

        h = F.relu(self.conv313(h))
        print (h.shape, gt_ab_313_va.shape)

        # h = self.class_reblance_layer.forward(h)
        loss = F.softmax_cross_entropy(h, gt_ab_313_va)

        chainer.report({"main/loss": loss})

        print (loss.data)

        return loss
