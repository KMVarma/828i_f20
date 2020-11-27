import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt

def preprocessed_rgb_to_ycbcr(rgb):
    integer_rgb = tf.cast(127.5*(rgb + 1), tf.uint8)
    integer_ycc = tfio.experimental.color.rgb_to_ycbcr(integer_rgb[0])
    float_ycc = tf.cast(integer_ycc, tf.float32)
    float_ycc = (float_ycc / 127.5)-1
    float_ycc = tf.expand_dims(float_ycc, axis=0)
    return float_ycc

def ypbpr_to_rgb(input):
    # inv of:
    # [[ 0.299   , 0.587   , 0.114   ],
    #  [-0.168736,-0.331264, 0.5     ],
    #  [ 0.5     ,-0.418688,-0.081312]]
    kernel = tf.constant(
        [
            [1.00000000e00, -1.21889419e-06, 1.40199959e00],
            [1.00000000e00, -3.44135678e-01, -7.14136156e-01],
            [1.00000000e00, 1.77200007e00, 4.06298063e-07],
        ],
        input.dtype,
    )
    return tf.tensordot(input, tf.transpose(kernel), axes=((-1,), (0,)))

def preprocessed_ycbcr_to_rgb(ycc):
    value = 127.5*(ycc + 1)
    value = value - tf.constant([16, 128, 128], value.dtype)
    value = value / tf.constant([219, 224, 224], value.dtype)
    value = ypbpr_to_rgb(value)
    value = 2 * value - 1
    return value

def imshow(im):
    # Convert [-1, 1] to [0, 1] range
    transformed_im = preprocessed_ycbcr_to_rgb(im)[0]*0.5 + 0.5
    transformed_im = tf.clip_by_value(transformed_im, 0, 1)
    return plt.imshow(transformed_im)
