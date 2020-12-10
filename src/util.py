import tensorflow as tf
import tensorflow_io as tfio

# Converts RGB pixels in [0,1] to YCC pixels in [0,1]
def rgb_to_ycc(rgb):
    integer_rgb = tf.cast(tf.math.floor(255.0 * rgb), tf.uint8)
    integer_ycc = tfio.experimental.color.rgb_to_ycbcr(integer_rgb)
    float_ycc = tf.cast(integer_ycc, tf.float32)
    float_ycc = float_ycc / 255.0
    float_ycc = tf.clip_by_value(float_ycc, 0, 1)

    return float_ycc


# Takes YPbPr pixels in [0, 255] and maps them to RGB pixels in [0,1]
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


# Converts a YCC-domain image with values in [0,1] to an RGB image with values in [0,1]
def ycc_to_rgb(ycc):
    value = 255.0 * ycc
    value = value - tf.constant([16, 128, 128], value.dtype)
    value = value / tf.constant([219, 224, 224], value.dtype)
    value = ypbpr_to_rgb(value)
    value = tf.clip_by_value(value, 0, 1)

    return value
