"""This is a TensorFlow modified implementation of AlexNet by Alex Krizhevsky et all.

This implementation has been adapted to work
with the small dimensions of the CIFAR-10 images (32 x 32).
Code for the original implementaion is also provided but has
been commented out.

Paper: ImageNet Classification with Deep Convolutional Neural Networks
(http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

Explanation on AlexNet can be found in my blog post:
https://mohitjain.me/2018/06/06/alexnet/

@author: Mohit Jain (contact: mohitjain1999(at)yahoo.com)

"""

import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()


def conv_layer(x, filter_height, filter_width,
               num_filters, stride, name, padding='SAME', groups=1):
    """Create a convolution layer."""

    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases of the conv layer
        W = tf.get_variable('weights', shape=[filter_height, filter_width,
                                              input_channels / groups, num_filters],
                            initializer=tf.random_normal_initializer(mean=0, stddev=0.01))

        # In the paper the biases of all of the layers have not been initialised the same way.
        # name[4] gives the number of the layer whose weights are being initialised.
        if (name[4] == '1' or name[4] == '3'):
            b = tf.get_variable('biases', shape=[num_filters], initializer=tf.constant_initializer(0.0))
        else:
            b = tf.get_variable('biases', shape=[num_filters], initializer=tf.constant_initializer(1.0))

    if groups == 1:
        conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

    # In the cases of multiple groups, split inputs & weights
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=W)

        output_groups = [tf.nn.conv2d(i, k, strides=[1, stride, stride, 1], padding=padding)
                         for i, k in zip(input_groups, weight_groups)]

        conv = tf.concat(axis=3, values=output_groups)

    # Add the biases.
    z = tf.nn.bias_add(conv, b)
    # Apply ReLu non linearity.
    a = tf.nn.relu(z, name=scope.name)

    return a


def fc_layer(x, input_size, output_size, name, relu=True):
    """Create a fully connected layer."""

    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases.
        W = tf.get_variable('weights', shape=[input_size, output_size],
                            initializer=tf.random_normal_initializer(mean=0, stddev=0.01))

        b = tf.get_variable('biases', shape=[output_size], initializer=tf.constant_initializer(1.0))

        # Matrix multiply weights and inputs and add biases.
        z = tf.nn.bias_add(tf.matmul(x, W), b, name=scope.name)

    if relu:
        # Apply ReLu non linearity.
        a = tf.nn.relu(z)
        return a

    else:
        return z


def max_pool(x, name, filter_height=3, filter_width=3, stride=2, padding='SAME'):
    """Create a max pooling layer."""

    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride, stride, 1], padding=padding,
                          name=name)


def lrn(x, name, radius=5, alpha=1e-04, beta=0.75, bias=2.0):
    """Create a local response normalization layer."""

    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha,
                                              beta=beta, bias=bias, name=name)


def dropout(x, keep_prob):
    """Create a dropout layer."""

    return tf.nn.dropout(x, keep_prob=keep_prob)


# Creating the AlexNet Model

class AlexNet(object):
    """Implementation of the AlexNet."""

    def __init__(self, x, keep_prob, num_classes):
        """Create the graph of the AlexNet model.

        Args:
            x: Placeholder for the input tensor.
            keep_prob: Dropout probability.
            num_classes: Number of classes in the dataset.
        """

        # Parse input arguments into class variables
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):
        """Create the network graph."""

        # In the original implementation this would be:
        # conv1 = conv_layer(self.X, 11, 11, 96, 4, padding = 'VALID', name = 'conv1')
        conv1 = conv_layer(self.X, 11, 11, 96, 2, name='conv1')
        norm1 = lrn(conv1, name='norm1')
        pool1 = max_pool(norm1, padding='VALID', name='pool1')

        conv2 = conv_layer(pool1, 5, 5, 256, 1, groups=2, name='conv2')
        norm2 = lrn(conv2, name='norm2')
        pool2 = max_pool(norm2, padding='VALID', name='pool2')

        conv3 = conv_layer(pool2, 3, 3, 384, 1, name='conv3')

        # This conv. layer has been removed in this modified implementation
        # but is present in the original paper implementaion.
        # conv4 = conv_layer(conv3, 3, 3, 384, 1, groups = 2, name = 'conv4')

        conv5 = conv_layer(conv3, 3, 3, 256, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, padding='VALID', name='pool5')

        # In the original paper implementaion this will be:
        # flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        # fc6 = fc_layer(flattened, 1 * 1 * 256, 4096, name = 'fc6')
        flattened = tf.reshape(pool5, [-1, 1 * 1 * 256])
        self.fc6 = fc_layer(flattened, 1 * 1 * 256, 1024, name='fc6')
        dropout6 = dropout(self.fc6, self.KEEP_PROB)

        # In the original paper implementaion this will be:
        # fc7 = fc_layer(dropout6, 4096, 4096, name = 'fc7')
        self.fc7 = fc_layer(dropout6, 1024, 2048, name='fc7')
        dropout7 = dropout(self.fc7, self.KEEP_PROB)

        # In the original paper implementaion this will be:
        # self.fc8 = fc_layer(dropout7, 4096, self.NUM_CLASSES, relu = False, name = 'fc8')
        self.fc8 = fc_layer(dropout7, 2048, self.NUM_CLASSES, relu=False, name='fc8')
