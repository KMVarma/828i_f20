import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src.utils import preprocessed_ycbcr_to_rgb, imshow

def predict_single(pretrained_model, im):
    return pretrained_model(preprocessed_ycbcr_to_rgb(im))

def create_chroma_adversarial_pattern(pretrained_model, input_image, input_label):
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = predict_single(pretrained_model, input_image)
        loss = loss_object(input_label, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image).numpy()
    # Clear the luma component of the gradient
    gradient[:,:,:,0] = 0
    # Normalize the gradient wrt the l-infty norm
    normalized_gradient = gradient / tf.norm(gradient, ord=np.inf)
    # Get the sign of the gradients to create the perturbation
    signed_gradient = tf.sign(gradient)
    return normalized_gradient

# DCT stuff was made with help from
# https://inst.eecs.berkeley.edu/~ee123/sp16/Sections/JPEG_DCT_Demo.html
# and https://fairyonice.github.io/2D-DCT.html

F = tf.signal.dct(tf.eye(8), type=2, norm="ortho")
FF = np.kron(F, F)
def dct2d(eight_by_eight_im):
    flattened_im = tf.reshape(eight_by_eight_im, (-1))
    flattened_res = FF @ tf.expand_dims(flattened_im, -1)
    return tf.reshape(flattened_res, eight_by_eight_im.shape)

bigF = tf.signal.dct(tf.eye(28), type=2, norm="ortho")
bigFF = np.kron(bigF, bigF)
def big_dct2d(im):
    flattened_im = tf.reshape(im, (-1))
    flattened_res = bigFF @ tf.expand_dims(flattened_im, -1)
    return tf.reshape(flattened_res, im.shape)

IF = tf.signal.dct(tf.eye(8), type=3, norm="ortho")
IFF = np.kron(IF, IF)
def idct2d(eight_by_eight_im):
    flattened_im = tf.reshape(eight_by_eight_im, (-1))
    flattened_res = IFF @ tf.expand_dims(flattened_im, -1)
    return tf.reshape(flattened_res, eight_by_eight_im.shape)

bigIF = tf.signal.dct(tf.eye(28), type=3, norm="ortho")
bigIFF = np.kron(bigIF, bigIF)
def big_idct2d(im):
    flattened_im = tf.reshape(im, (-1))
    flattened_res = bigIFF @ tf.expand_dims(flattened_im, -1)
    return tf.reshape(flattened_res, im.shape)

# Expects an image of the form (w, h) where each entry is in [0, 255]
def blockwise_monochannel_dct(im):
    imsize = im.shape
    out = tf.Variable(tf.zeros(imsize))
    # Do 8x8 DCT on image (in-place)
    for i in range(0, imsize[0], 8):
        for j in range(0, imsize[1], 8):
            out[i:(i+8),j:(j+8)].assign(dct2d(im[i:(i+8),j:(j+8)]))
    return out

# Expects an image of the form (w, h) where each entry is in [0, 255]
def big_blockwise_monochannel_dct(im):
    imsize = im.shape
    out = tf.Variable(tf.zeros(imsize))
    # Do 28x28 DCT on image (in-place)
    for i in range(0, imsize[0], 28):
        for j in range(0, imsize[1], 28):
            out[i:(i+28),j:(j+28)].assign(big_dct2d(im[i:(i+28),j:(j+28)]))
    return out

def blockwise_dct_channels(im):
    # Do DCT for Y, Cb, and Cr separately
    y_dct = blockwise_monochannel_dct(im[:,:,0])
    cb_dct = blockwise_monochannel_dct(im[:,:,1])
    cr_dct = blockwise_monochannel_dct(im[:,:,2])

    return y_dct, cb_dct, cr_dct

def blockwise_dct(im):
    y_dct, cb_dct, cr_dct = blockwise_dct_channels(im)
    return tf.stack([y_dct, cb_dct, cr_dct], axis=-1)

def big_blockwise_dct_channels(im):
    # Do DCT for Y, Cb, and Cr separately
    y_dct = big_blockwise_monochannel_dct(im[:,:,0])
    cb_dct = big_blockwise_monochannel_dct(im[:,:,1])
    cr_dct = big_blockwise_monochannel_dct(im[:,:,2])

    return y_dct, cb_dct, cr_dct

def big_blockwise_dct(im):
    y_dct, cb_dct, cr_dct = big_blockwise_dct_channels(im)
    return tf.stack([y_dct, cb_dct, cr_dct], axis=-1)

def blockwise_monochannel_idct(imhat):
    imsize = imhat.shape
    out = tf.Variable(tf.zeros(imsize))
    # Do 8x8 IDCT on image (in-place)
    for i in range(0, imsize[0], 8):
        for j in range(0, imsize[1], 8):
            out[i:(i+8),j:(j+8)].assign(idct2d(imhat[i:(i+8),j:(j+8)]))
    return out

def big_blockwise_monochannel_idct(imhat):
    imsize = imhat.shape
    out = tf.Variable(tf.zeros(imsize))
    # Do 28x28 IDCT on image (in-place)
    rows = []
    for i in range(0, imsize[0], 28):
        row = []
        for j in range(0, imsize[1], 28):
            row.append(big_idct2d(imhat[i:(i+28),j:(j+28)]))

        concatted_row = tf.concat(row, axis=1)
        rows.append(concatted_row)

    return tf.concat(rows, axis=0)

quant_mat = tf.constant(\
[[16, 11, 10, 16, 24, 40, 51, 61],
[12, 12, 14, 19, 26, 58, 60, 55],
[14, 13, 16, 24, 40, 57, 69, 56],
[14, 17, 22, 29, 51, 87, 80, 62],
[18, 22, 37, 56, 68, 109, 103, 77],
[24, 35, 55, 64, 81, 104, 113, 92],
[49, 64, 78, 87, 103, 121, 120, 101],
[72, 92, 95, 98, 112, 100, 103, 99]], \
dtype=tf.float32)

#quant_mat = tf.constant(\
#[[1, 1, 1, 1, 1, 1, 1, 1],
#[1, 1, 1, 1, 1, 1, 1, 1],
#[1, 1, 1, 1, 1, 1, 1, 1],
#[1, 1, 1, 1, 1, 1, 1, 1],
#[1, 1, 1, 1, 1, 1, 1, 1],
#[1, 1, 1, 1, 1, 100, 1, 1],
#[1, 1, 100, 1, 100, 1, 1, 1],
#[1, 1, 100, 1, 1, 1, 1, 100]],\
#dtype=tf.float32)

def quantize_freq(imhat, q):
    return tf.round(imhat / q) * q

def create_freq_adversarial_pattern(pretrained_model, input_image, input_label):
    loss_object = tf.keras.losses.CategoricalCrossentropy()

    scaled_input = input_image[0]*127.5 - 0.5

    pos = 12*8
    imhat_y, imhat_cb, imhat_cr  = blockwise_dct_channels(scaled_input)
    tf.print("imhat_y is", imhat_y[pos:pos+8,pos:pos+8], summarize=-1)
    print("imhat_y shape is ", imhat_y.shape)
    print("imhat_y min is ", tf.reduce_min(imhat_y))
    print("imhat_y max is ", tf.reduce_max(imhat_y))

    plt.figure()
    plt.imshow(tf.cast(scaled_input + 128, tf.uint8),cmap='gray')
    plt.title("Image")

    # Display the dct of that block
    plt.figure()
    normalized_imhat = imhat_y / tf.norm(imhat_y, ord=np.inf)
    plt.imshow(imhat_y*0.5 + 0.5,cmap='gray')
    plt.title( "DCT of whole image")


    im_y = tf.cast(blockwise_monochannel_idct(imhat_y) + 128, tf.uint8)
    tiled_quant_mat = tf.tile(quant_mat, (tf.constant(imhat_y.shape) // 8))
    for eps in [0.01, 0.05, 0.1, 0.5, 1]:
        quantized_imhat_y = quantize_freq(imhat_y, eps * tiled_quant_mat)
        compressed_im_y = tf.cast(blockwise_monochannel_idct(quantized_imhat_y) + 128, tf.uint8)
        plt.figure(figsize=(20,10))
        plt.imshow(tf.concat([im_y, compressed_im_y], axis=1),cmap='gray')
        plt.title(f"Image round trip with eps == {eps:.2f}")

    imhat = big_blockwise_dct(scaled_input)
    with tf.GradientTape() as tape:
        tape.watch([imhat_y, imhat_cb, imhat_cr])
        im_y = (big_blockwise_monochannel_idct(imhat_y) + 0.5) / 127.5
        im_cb = (big_blockwise_monochannel_idct(imhat_cb) + 0.5) / 127.5
        im_cr = (big_blockwise_monochannel_idct(imhat_cr) + 0.5) / 127.5
        #im_y = (blockwise_monochannel_idct(imhat_y) + 0.5) / 127.5
        #im_cb = (blockwise_monochannel_idct(imhat_cb) + 0.5) / 127.5
        #im_cr = (blockwise_monochannel_idct(imhat_cr) + 0.5) / 127.5
        stacked = tf.expand_dims(tf.stack([im_y, im_cb, im_cr], axis=-1), axis=0)
        print("stacked shape", stacked.shape)
        prediction = predict_single(pretrained_model, stacked)
        loss = loss_object(input_label, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient_y, gradient_cb, gradient_cr = tape.gradient(loss, [imhat_y, imhat_cb, imhat_cr])
    gradient = tf.stack([gradient_y, gradient_cb, gradient_cr], axis=-1).numpy()
    # Normalize the gradient wrt the l-infty norm
    #normalized_gradient = gradient * tf.stack([tiled_quant_mat, tiled_quant_mat, tiled_quant_mat], axis=-1)
    normalized_gradient = gradient / tf.norm(gradient, ord=np.inf)
    # Get the sign of the gradients to create the perturbation
    signed_gradient = tf.sign(gradient)
    return tf.expand_dims(normalized_gradient, axis=0)
