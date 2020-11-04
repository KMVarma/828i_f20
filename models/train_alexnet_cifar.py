import tensorflow.compat.v1 as tf

import numpy as np
from alexnet import AlexNet
import os
import pdb

tf.disable_v2_behavior()

"""
x train (50000, 32, 32, 3)
y train (50000, 1)
x test (10000, 32, 32, 3)
y test (10000, 1)
"""
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# placeholder for input and dropout rate
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_true = tf.placeholder(tf.float32, shape=[None, 10])   # Tensor 'Placeholder_1:0'
keep_prob = tf.placeholder(tf.float32)

# Create the AlexNet model
model = AlexNet(x=x, keep_prob=keep_prob, num_classes=10)

# define activation of last layer as score
# Tensor("fc8/fc8:0", shape=(?, 2), dtype=float32)
score = model.fc8

# loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=score))

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(cross_entropy)

# Initialize all global variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()
i = 0


def next_batch(batch_size):
    # Note that the 100 dimension in the reshape call is set by an assumed batch size of 100
    global i
    x = x_train[i:i + batch_size].reshape(batch_size, 32, 32, 3)
    y = y_train[i:i + batch_size]
    i = (i + batch_size) % len(x_train)
    return x, y


# steps = 10,000
# about 10,000 in training set
# (10,000 * 100) / 10,000 = 100 epochs
steps = 10000
saved_feats = []

with tf.Session() as sess:
    sess.run(init)

    for i in range(steps):

        # get next batch of data.
        try:
            batch = next_batch(100)
        except:
            print('caught')  # this happened 138 times
            continue
        # On training set.
        sess.run(train, feed_dict={x: batch[0], y_true: batch[1], keep_prob: 0.5})

        # Print accuracy after every epoch.
        # 500 * 100 = 50,000 which is one complete batch of data.
        if i % 500 == 0:
            print("EPOCH: {}".format(i / 500))
            print("ACCURACY ")

            matches = tf.equal(tf.argmax(score, 1), tf.argmax(y_true, 1))
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))

            # On valid/test set. (this is getting the accuracy)
            print(sess.run(acc, feed_dict={x: x_test, y_true: y_test, keep_prob: 1.0}))
            print('\n')

    if not os.path.exists('./models/saved/alexnet/'):
        os.makedirs('./models/saved/alexnet/')
    saver.save(sess, './models/saved/alexnet/model')
"""
x_tr = np.load('./data/x_tr.npy')
x_test = np.load('./data/x_tst.npy')

y_tr = np.load('./data/y_tr.npy')
y_test = np.load('./data/y_tst.npy')

# placeholder for input and dropout rate
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_true = tf.placeholder(tf.float32, shape=[None, 2])
keep_prob = tf.placeholder(tf.float32)

# Create the AlexNet model
model = AlexNet(x=x, keep_prob=keep_prob, num_classes=2)

# define activation of last layer as score
# Tensor("fc8/fc8:0", shape=(?, 2), dtype=float32)
score = model.fc8

# activations of penultimate layer
# <tf.Tensor 'Relu_1:0' shape=(?, 2048) dtype=float32>
activations = model.fc7

# features
# <tf.Tensor 'Relu:0' shape=(?, 1024) dtype=float32>
features = model.fc6

# loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=score))

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(cross_entropy)

# Initialize all global variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()
i = 0


def next_batch(batch_size):
    # Note that the 100 dimension in the reshape call is set by an assumed batch size of 100
    global i
    x = x_tr[i:i + batch_size].reshape(100, 32, 32, 3)
    y = y_tr[i:i + batch_size]
    i = (i + batch_size) % len(x_tr)
    return x, y


# steps = 10,000
# about 10,000 in training set
# (10,000 * 100) / 10,000 = 100 epochs
steps = 10000
saved_feats = []

with tf.Session() as sess:
    sess.run(init)

    for i in range(steps):

        # get next batch of data.
        try:
            batch = next_batch(100)
        except:
            print('caught')  # this happened 138 times
            continue
        # On training set.
        sess.run(train, feed_dict={x: batch[0], y_true: batch[1], keep_prob: 0.5})

        # Print accuracy after every epoch.
        # 500 * 100 = 50,000 which is one complete batch of data.
        if i % 500 == 0:
            print("EPOCH: {}".format(i / 500))
            print("ACCURACY ")

            matches = tf.equal(tf.argmax(score, 1), tf.argmax(y_true, 1))
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))

            # On valid/test set. (this is getting the accuracy)
            print(sess.run(acc, feed_dict={x: x_test, y_true: y_test, keep_prob: 1.0}))
            print('\n')

    test_feats = features.eval(session=sess, feed_dict={x: x_test, y_true: y_test, keep_prob: 1.0})
    print(np.shape(test_feats))
    print('\n')

    train_feats = features.eval(session=sess, feed_dict={x: x_tr, y_true: y_tr, keep_prob: 1.0})
    print(np.shape(train_feats))
    print('\n')

    test_activations = activations.eval(session=sess, feed_dict={x: x_test, y_true: y_test, keep_prob: 1.0})
    print(np.shape(test_activations))
    print('\n')

    train_activations = activations.eval(session=sess, feed_dict={x: x_tr, y_true: y_tr, keep_prob: 1.0})
    print(np.shape(train_activations))
    print('\n')

    if not os.path.exists('./poisoning/alexnet/'):
        os.makedirs('./poisoning/alexnet/')
    saver.save(sess, './poisoning/alexnet/model')

x_tr_feats = np.array(train_feats)
np.save('./poisoning/Data/x_tr_f.npy', x_tr_feats)

x_tst_feats = np.array(test_feats)
np.save('./poisoning/Data/x_tst_f.npy', x_tst_feats)

x_tr_activations = np.array(train_activations)
np.save('./poisoning/Data/x_tr_act.npy', x_tr_activations)

x_tst_activations = np.array(test_activations)
np.save('./poisoning/Data/x_tst_act.npy', x_tst_activations)
"""