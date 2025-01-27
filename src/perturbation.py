# Copyright from tensorflow's official FGSM tutorial
# (https://www.tensorflow.org/tutorials/generative/adversarial_fgsm)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .dct import blockwise_dct, blockwise_idct
from .obfuscation import obfuscate_freq_gradient, obfuscate_chroma_gradient
from .util import ycc_to_rgb

from math import inf

import tensorflow as tf

# Inputs:
#   * im: An image of dimension [H, W, 3] with YCC values in [0, 1]
#   * gt_class: The (integer) ground-truth class of the image being classified
#   * classify_f: The classification function we're trying to fool. This function should take images
#     of dim [H, W, 3] with RGB values in [0, 1]
# Output:
#   * Two l-infty normalized adversarial gradients of dimension [H, W, 3] with YCC values in [0, 1]
def adv_chroma_and_freq_gradient(im, gt_class, classify_f, dct_blocksize=8):
    # Use cross entropy loss
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    # Shift the image coefficients to the range [-128, 127]
    scaled_input = im * 255.0 - 128.0
    # Convert the image to DCT form
    imhat = blockwise_dct(scaled_input, dct_blocksize)

    # Compute the gradient of the loss function on this input
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(imhat)
        # Convert the image back to YCC values in the range [-128, 127]
        ycc_im = blockwise_idct(imhat, dct_blocksize)
        # Shift it back to [0, 1] range
        shifted_ycc_im = (ycc_im + 128.0) / 255.0
        # Convert to RGB
        rgb_im = ycc_to_rgb(shifted_ycc_im)

        # Make the prediction and compute the loss
        prediction = classify_f(rgb_im)
        loss = loss_object([gt_class], prediction)

    # Get the gradients of the loss w.r.t to the YCC input image
    chroma_grad = tape.gradient(loss, shifted_ycc_im).numpy()
    # Obfuscate the gradient so it's less perceptible to the human eye
    chroma_grad = obfuscate_chroma_gradient(chroma_grad)
    # Normalize the gradient wrt the l-infty norm
    chroma_grad = tf.linalg.normalize(chroma_grad, ord=inf)[0]

    # Get the gradients of the loss w.r.t to the DCT of input image.
    freq_grad = tape.gradient(loss, imhat).numpy()
    # Obfuscate the gradient so it's less perceptible to the human eye
    obfuscated_freq_grad = obfuscate_freq_gradient(freq_grad, dct_blocksize)
    # Now convert the gradient back to color space and normalize it
    obfuscated_freq_grad = blockwise_idct(obfuscated_freq_grad, dct_blocksize)
    obfuscated_freq_grad = tf.linalg.normalize(obfuscated_freq_grad, ord=inf)[0]

    return chroma_grad, obfuscated_freq_grad
