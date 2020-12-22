import tensorflow as tf
from tensorflow.image import ResizeMethod

# Takes a [H, W, 3] YCC image with values in [0,1] and outputs the "obfuscated" version of it. All
# it actually does is clear the luma component, because change in luma is the most perceptible to
# human eyes.
def obfuscate_chroma_gradient(grad):
    # Luma component is the Y in YCC. Set it to 0
    grad[:, :, 0] = 0
    # To max out the l-infty weight of the gradient, take the sign
    grad = tf.sign(grad)

    return grad


# Quantization matrices for Y channel and Cr/Cb channels
# Source: https://www.sciencedirect.com/topics/computer-science/quantization-matrix

q_y = tf.constant(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=tf.float32,
)

q_chroma = tf.constant(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=tf.float32,
)

# Takes a tensor of dimension [H, W, 3] and "obfuscates" its more human-perceptible frequencies.
# Basically, this just weakens the coefficients of the longer frequencies and strengthens the
# coefficients of the shorter frequencies. The quantization matrices encode which frequencies are
# more perceptible to the human eye for 8x8 DCT blocks. We generalize this by assuming that the
# quantization matrices can be scaled up and down for different choices of DCT block size. It's
# unclear if this is a valid thing to assume.
def obfuscate_freq_gradient(gradient, dct_blocksize):
    channel_shape = tf.constant(gradient[:, :, 0].shape)
    dct_block_shape = tf.constant([dct_blocksize, dct_blocksize])

    # First scale the quantization matrices to fit our DCT blocksize. There's no reason to believe
    # that they generalize like this, but who cares
    scaled_q_y = tf.image.resize(
        q_y[..., tf.newaxis], dct_block_shape, method=ResizeMethod.BILINEAR
    )
    scaled_q_chroma = tf.image.resize(
        q_chroma[..., tf.newaxis], dct_block_shape, method=ResizeMethod.BILINEAR
    )

    # Remove the extra (empty) dimension the scaling added
    scaled_q_y = tf.reshape(scaled_q_y, dct_block_shape)
    scaled_q_chroma = tf.reshape(scaled_q_chroma, dct_block_shape)

    # Now tile the quantization matrix to the size of the whole image
    tiled_q_y = tf.tile(scaled_q_y, (channel_shape // dct_blocksize))
    tiled_q_chroma = tf.tile(scaled_q_chroma, (channel_shape // dct_blocksize))

    # Combine the quantization matrices from all the channels
    q = tf.stack([tiled_q_y, tiled_q_chroma, tiled_q_chroma], axis=-1)


    # Use the quantization matrix as weights. The more human-perceptible frequencies have lower
    # weights than the less human-perceptible frequencies. The sign() is to maximize the l-infty
    # weight wrt the weight matrix
    return gradient * q
