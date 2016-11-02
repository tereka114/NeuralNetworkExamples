# coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
# **************************************
# ***** Richard Zhang / 2016.08.06 *****
# **************************************
import numpy as np
import warnings
import os
import sklearn.neighbors as nn
from skimage import color


# class BGR2LabLayer(object):
#     ''' Layer converts BGR to Lab
#     INPUTS
#         bottom[0].data  Nx3xXxY
#     OUTPUTS
#         top[0].data     Nx3xXxY
#     '''
#     def __init__(self, N, X, Y):
#         self.N = N
#         self.X = X
#         self.Y = Y
#
#     def reshape(self, bottom, top):
#         top[0].reshape(self.N, 3, self.X, self.Y)
#
#     def forward(self, bottom, top):
#         top[0].data[...] = color.rgb2lab(
#             bottom[0].data[:, ::-1, :, :].astype('uint8').transpose((2, 3, 0, 1))).transpose((2, 3, 0, 1))


class NNEncLayer(object):
    ''' Layer which encodes ab map into Q colors
    INPUTS
        bottom[0]   Nx2xXxY
    OUTPUTS
        top[0].data     NxQ
    '''
    def __init__(self, NN, sigma, N, X, Y, Q):
        self.NN = NN
        self.sigma = sigma
        self.ENC_DIR = './resources/'
        self.nnenc = NNEncode(self.NN, self.sigma, km_filepath=os.path.join(self.ENC_DIR, 'pts_in_hull.npy'))

        self.N = N
        self.X = X
        self.Y = Y
        self.Q = self.nnenc.K

    def forward(self, x):
        return self.nnenc.encode_points_mtx_nd(x)

    def reshape(self, bottom, top):
        top[0].reshape(self.N, self.Q, self.X, self.Y)


class PriorBoostLayer(object):
    ''' Layer boosts ab values based on their rarity
    INPUTS
        bottom[0]       NxQxXxY
    OUTPUTS
        top[0].data     Nx1xXxY
    '''

    def __init__(self, bottom, top, ENC_DIR, gamma, alpha, N, Q, X, Y):
        if len(bottom) == 0:
            raise Exception("Layer should have inputs")

        self.ENC_DIR = './resources/'
        self.gamma = .5
        self.alpha = 1.
        self.pc = PriorFactor(self.alpha, gamma=self.gamma, priorFile=os.path.join(self.ENC_DIR, 'prior_probs.npy'))

        self.N = N
        self.Q = Q
        self.X = X
        self.Y = Y

    def reshape(self, bottom, top):
        top[0].reshape(self.N, 1, self.X, self.Y)

    def forward(self, bottom, top):
        top[0].data[...] = self.pc.forward(bottom[0].data[...], axis=1)


class NonGrayMaskLayer(object):
    ''' Layer outputs a mask based on if the image is grayscale or not
    INPUTS
        bottom[0]       Nx2xXxY     ab values
    OUTPUTS
        top[0].data     Nx1xXxY     1 if image is NOT grayscale
                                    0 if image is grayscale
    '''

    def setup(self, bottom, top):
        if len(bottom) == 0:
            raise Exception("Layer should have inputs")

        self.thresh = 5  # threshold on ab value
        self.N = bottom[0].data.shape[0]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]

    def reshape(self, bottom, top):
        top[0].reshape(self.N, 1, self.X, self.Y)

    def forward(self, bottom, top):
        # if an image has any (a,b) value which exceeds threshold, output 1
        top[0].data[...] = (np.sum(np.sum(np.sum(np.abs(bottom[0].data) > self.thresh, axis=1), axis=1), axis=1) > 0)[:,
                           na(), na(), na()]


class ClassRebalanceMultLayer(object):
    ''' INPUTS
        bottom[0]   NxMxXxY     feature map
        bottom[1]   Nx1xXxY     boost coefficients
    OUTPUTS
        top[0]      NxMxXxY     on forward, gets copied from bottom[0]
    FUNCTIONALITY
        On forward pass, top[0] passes bottom[0]
        On backward pass, bottom[0] gets boosted by bottom[1]
        through pointwise multiplication (with singleton expansion) '''
    def reshape(self, bottom, top):
        i = 0
        if (bottom[i].data.ndim == 1):
            top[i].reshape(bottom[i].data.shape[0])
        elif (bottom[i].data.ndim == 2):
            top[i].reshape(bottom[i].data.shape[0], bottom[i].data.shape[1])
        elif (bottom[i].data.ndim == 4):
            top[i].reshape(bottom[i].data.shape[0], bottom[i].data.shape[1], bottom[i].data.shape[2],
                           bottom[i].data.shape[3])

    def forward(self, x):
        # output equation to negative of inputs
        # top[0].data[...] = bottom[0].data[...]
        return x
        # top[0].data[...] = bottom[0].data[...]*bottom[1].data[...] # this was bad, would mess up the gradients going up

    # def backward(self, top, propagate_down, bottom):
    #     for i in range(len(bottom)):
    #         if not propagate_down[i]:
    #             continue
    #         bottom[0].diff[...] = top[0].diff[...] * bottom[1].data[...]
            # print 'Back-propagating class rebalance, %i'%i


# ***************************
# ***** SUPPORT CLASSES *****
# ***************************
class PriorFactor():
    ''' Class handles prior factor '''

    def __init__(self, alpha, gamma=0, verbose=True, priorFile=''):
        # INPUTS
        #   alpha           integer     prior correction factor, 0 to ignore prior, 1 to divide by prior, alpha to divide by prior**alpha
        #   gamma           integer     percentage to mix in uniform prior with empirical prior
        #   priorFile       file        file which contains prior probabilities across classes

        # settings
        self.alpha = alpha
        self.gamma = gamma
        self.verbose = verbose

        # empirical prior probability
        self.prior_probs = np.load(priorFile)

        # define uniform probability
        self.uni_probs = np.zeros_like(self.prior_probs)
        self.uni_probs[self.prior_probs != 0] = 1.
        self.uni_probs = self.uni_probs / np.sum(self.uni_probs)

        # convex combination of empirical prior and uniform distribution
        self.prior_mix = (1 - self.gamma) * self.prior_probs + self.gamma * self.uni_probs

        # set prior factor
        self.prior_factor = self.prior_mix ** -self.alpha
        self.prior_factor = self.prior_factor / np.sum(self.prior_probs * self.prior_factor)  # re-normalize

        # implied empirical prior
        self.implied_prior = self.prior_probs * self.prior_factor
        self.implied_prior = self.implied_prior / np.sum(self.implied_prior)  # re-normalize

        if (self.verbose):
            self.print_correction_stats()

    def print_correction_stats(self):
        print('Prior factor correction:')
        print('  (alpha,gamma) = (%.2f, %.2f)' % (self.alpha, self.gamma))
        print('  (min,max,mean,med,exp) = (%.2f, %.2f, %.2f, %.2f, %.2f)' % (
            np.min(self.prior_factor), np.max(self.prior_factor), np.mean(self.prior_factor),
            np.median(self.prior_factor),
            np.sum(self.prior_factor * self.prior_probs)))

    def forward(self, data_ab_quant, axis=1):
        data_ab_maxind = np.argmax(data_ab_quant, axis=axis)
        corr_factor = self.prior_factor[data_ab_maxind]
        if (axis == 0):
            return corr_factor[na(), :]
        elif (axis == 1):
            return corr_factor[:, na(), :]
        elif (axis == 2):
            return corr_factor[:, :, na(), :]
        elif (axis == 3):
            return corr_factor[:, :, :, na()]


class NNEncode():
    ''' Encode points using NN search and Gaussian kernel '''

    def __init__(self, NN, sigma, km_filepath='', cc=-1):
        if (check_value(cc, -1)):
            self.cc = np.load(km_filepath)
        else:
            self.cc = cc
        self.K = self.cc.shape[0]
        self.NN = int(NN)
        self.sigma = sigma
        self.nbrs = nn.NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(self.cc)

        self.alreadyUsed = False

    def encode_points_mtx_nd(self, pts_nd, axis=1, returnSparse=False, sameBlock=True):
        pts_flt = flatten_nd_array(pts_nd, axis=axis)
        P = pts_flt.shape[0]
        if (sameBlock and self.alreadyUsed):
            self.pts_enc_flt[...] = 0  # already pre-allocated
        else:
            self.alreadyUsed = True
            self.pts_enc_flt = np.zeros((P, self.K))
            self.p_inds = np.arange(0, P, dtype='int')[:, na()]

        P = pts_flt.shape[0]

        (dists, inds) = self.nbrs.kneighbors(pts_flt)

        wts = np.exp(-dists ** 2 / (2 * self.sigma ** 2))
        wts = wts / np.sum(wts, axis=1)[:, na()]

        self.pts_enc_flt[self.p_inds, inds] = wts
        pts_enc_nd = unflatten_2d_array(self.pts_enc_flt, pts_nd, axis=axis)

        return pts_enc_nd


# *****************************
# ***** Utility functions *****
# *****************************
def check_value(inds, val):
    ''' Check to see if an array is a single element equaling a particular value
    for pre-processing inputs in a function '''
    if (np.array(inds).size == 1):
        if (inds == val):
            return True
    return False


def na():  # shorthand for new axis
    return np.newaxis


def flatten_nd_array(pts_nd, axis=1):
    ''' Flatten an nd array into a 2d array with a certain axis
    INPUTS
        pts_nd       N0xN1x...xNd array
        axis         integer
    OUTPUTS
        pts_flt     prod(N \ N_axis) x N_axis array     '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0, NDIM), np.array((axis)))  # non axis indices
    NPTS = np.prod(SHP[nax])
    axorder = np.concatenate((nax, np.array(axis).flatten()), axis=0)
    pts_flt = pts_nd.transpose((axorder))
    pts_flt = pts_flt.reshape(NPTS, SHP[axis])
    return pts_flt


def unflatten_2d_array(pts_flt, pts_nd, axis=1, squeeze=False):
    ''' Unflatten a 2d array with a certain axis
    INPUTS
        pts_flt     prod(N \ N_axis) x M array
        pts_nd      N0xN1x...xNd array
        axis        integer
        squeeze     bool     if true, M=1, squeeze it out
    OUTPUTS
        pts_out     N0xN1x...xNd array        '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0, NDIM), np.array((axis)))  # non axis indices

    if (squeeze):
        axorder = nax
        axorder_rev = np.argsort(axorder)
        NEW_SHP = SHP[nax].tolist()

        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)
    else:
        axorder = np.concatenate((nax, np.array(axis).flatten()), axis=0)
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        NEW_SHP.append(M)
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)

    return pts_out
