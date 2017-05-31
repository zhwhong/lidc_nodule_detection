import os

import numpy as np
import scipy.io
import tensorflow as tf
import tflearn

_data = None


def load_mat():
    global _data
    if _data is not None:
        return _data
    try_path = ['.', '/tmp']
    filename = 'imagenet-vgg-verydeep-19.mat'
    for dir in try_path:
        path = os.path.join(dir, filename)
        if os.path.isfile(path):
            _data = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')
            return _data
    raise IOError("Network %s not found." % filename)


def net(input_image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4', # 'pool5'
    )

    global _data
    if _data is None:
        _data = load_mat()
    weights = _data['layers'][0]

    with tf.name_scope('vgg'):
        network = input_image
        for i, name in enumerate(layers):
            with tf.name_scope('vgg_layer_%s' % name):
                kind = name[:4]
                if kind == 'conv':
                    kernels, bias = weights[i][0][0][0][0]
                    # matconvnet: weights are [width, height, in_channels, out_channels]
                    # tensorflow: weights are [height, width, in_channels, out_channels]
                    kernels = np.transpose(kernels, (1, 0, 2, 3))
                    bias = bias.reshape(-1)
                    conv = tf.nn.conv2d(network, tf.constant(kernels),
                                        strides=(1, 1, 1, 1), padding='SAME',
                                        name=name)
                    network = tf.nn.bias_add(conv, bias)
                elif kind == 'relu':
                    network = tf.nn.relu(network, name=name)
                elif kind == 'pool':
                    network = tf.nn.max_pool(network, ksize=(1, 2, 2, 1),
                                             strides=(1, 2, 2, 1),
                                             padding='SAME')
    return network


def classifier():
    input_image = tflearn.input_data([None, 64, 64, 3])
    vgg = net(input_image)
    network = tflearn.layers.flatten(vgg)
    # network = tflearn.fully_connected(network, 4096, activation='relu')
    # network = tflearn.fully_connected(network, 4096, activation='relu')
    # network = tflearn.fully_connected(network, 2)
    # network = tflearn.fully_connected(network, 2, activation='sigmoid')
    x = network
    _, n_features = map(lambda v: v.value, x.get_shape())
    weights = tf.Variable(tf.random_normal([n_features, 2]))
    bias = tf.Variable(tf.random_normal([2]))
    network = tf.nn.softmax(tf.matmul(x, weights) + bias)
    return tflearn.regression(network)


# model = tflearn.DNN(classifier(),tensorboard_dir="./tmp/tflearn_logs/")
model = tflearn.DNN(classifier(), tensorboard_verbose=1, tensorboard_dir="./tmp/tflearn_logs/")

