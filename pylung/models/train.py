import csv
import os

import easytf
import tensorflow as tf
from pylung.models import vgg_classifier
from pylung.utils import dcm_to_gray, gray_to_rgb
from scipy.misc import imread

tf.app.flags.DEFINE_integer('n_epoch', 10, 'epoch number')
FLAGS = tf.app.flags.FLAGS


def load_data(batch=None):
    csvfile = file('test.csv', 'rb')  # control train set
    reader = csv.reader(csvfile)
    images, labels = [], []
    count = 0
    for line in reader:
        image = imread(line[0] + line[1])
        images.append(gray_to_rgb(dcm_to_gray(image)))
        y = int(line[2])
        labels.append([1 - y, y])
        count += 1
    return images, labels


def train(batch=0, n_epoch=10, checkpoint=None):
    images, labels = load_data(batch)
    model = easytf.load_model_by_module(vgg_classifier)
    if checkpoint is not None and os.path.isfile(checkpoint):
        model.load(checkpoint)
    model.fit(images, labels, n_epoch=n_epoch)
    if checkpoint is not None:
        model.save(checkpoint)


def main():
    train()


if __name__ == '__main__':
    main()
