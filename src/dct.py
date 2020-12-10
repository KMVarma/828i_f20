# DCT stuff was made with help from
# https://inst.eecs.berkeley.edu/~ee123/sp16/Sections/JPEG_DCT_Demo.html
# and https://fairyonice.github.io/2D-DCT.html

from functools import lru_cache

import numpy as np
import tensorflow as tf

# Constructs a 2D-DCT matrix with side length of block_size. This is cached so the big matrix
# doesn't have to be recomputed every call
@lru_cache
def dct_mat(block_size):
    m = tf.signal.dct(tf.eye(block_size), type=2, norm="ortho")
    mm = np.kron(m, m)
    return mm


# Constructs a 2D-IDCT matrix with side length of block_size. This is cached so the big matrix
# doesn't have to be recomputed every call
@lru_cache
def idct_mat(block_size):
    m = tf.signal.dct(tf.eye(block_size), type=3, norm="ortho")
    mm = np.kron(m, m)
    return mm


# Performs a 2D-DCT on an input tensor of shape [H, W, 3], where H = W (i.e., it's square)
def dct2d(subimage):
    mm = dct_mat(subimage.shape[0])
    flattened_im = tf.reshape(subimage, (-1))
    flattened_res = mm @ flattened_im[..., tf.newaxis]
    return tf.reshape(flattened_res, subimage.shape)


# Performs a 2D-IDCT on an input tensor of shape [H, W, 3], where H = W (i.e., it's square)
def idct2d(subimage_hat):
    mm = idct_mat(subimage_hat.shape[0])
    flattened_imhat = tf.reshape(subimage_hat, (-1))
    flattened_res = mm @ flattened_imhat[..., tf.newaxis]
    return tf.reshape(flattened_res, subimage_hat.shape)


# Performs a blockwise 2D-DCT on an input tensor of shape [H, W], where H = W and the side length of
# each DCT block is block_size
def blockwise_monochannel_dct(im, block_size):
    imsize = im.shape
    rows = []

    for i in range(0, imsize[0], block_size):
        row = []
        for j in range(0, imsize[1], block_size):
            row.append(dct2d(im[i : (i + block_size), j : (j + block_size)]))

        concatted_row = tf.concat(row, axis=1)
        rows.append(concatted_row)

    return tf.concat(rows, axis=0)


# Performs a blockwise 2D-IDCT on an input tensor of shape [H, W], where H = W and the side length
# of each IDCT block is block_size
def blockwise_monochannel_idct(imhat, block_size):
    imsize = imhat.shape
    rows = []
    for i in range(0, imsize[0], block_size):
        row = []
        for j in range(0, imsize[1], block_size):
            row.append(idct2d(imhat[i : (i + block_size), j : (j + block_size)]))

        concatted_row = tf.concat(row, axis=1)
        rows.append(concatted_row)

    return tf.concat(rows, axis=0)


# Performs a blockwise 2D-DCT on an input tensor of shape [H, W, 3], where H = W and the side
# length of each DCT block is block_size
def blockwise_dct(im, block_size):
    # Do DCT for each channel separately
    chan0_dct = blockwise_monochannel_dct(im[:, :, 0], block_size)
    chan1_dct = blockwise_monochannel_dct(im[:, :, 1], block_size)
    chan2_dct = blockwise_monochannel_dct(im[:, :, 2], block_size)

    return tf.stack([chan0_dct, chan1_dct, chan2_dct], axis=-1)


# Performs a blockwise 2D-IDCT on an input tensor of shape [H, W, 3], where H = W and the side
# length of each IDCT block is block_size
def blockwise_idct(imhat, block_size):
    # Do IDCT for each channel separately
    chan0 = blockwise_monochannel_idct(imhat[:, :, 0], block_size)
    chan1 = blockwise_monochannel_idct(imhat[:, :, 1], block_size)
    chan2 = blockwise_monochannel_idct(imhat[:, :, 2], block_size)

    return tf.stack([chan0, chan1, chan2], axis=-1)
