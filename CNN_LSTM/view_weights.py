#!/usr/bin/env python
#-*-coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
from scipy.misc import imsave
from PIL import Image
import cv2


def find_all_files(root, suffix=None):
    res = []
    for root, _, files in os.walk(root):
        for f in files:
            if suffix is not None and not f.endswith(suffix):
                continue
            res.append(os.path.join(root, f))
    return res


def main():
    checkpoint_dir = './data/inception_v1.ckpt'
    reader = tf.train.NewCheckpointReader(checkpoint_dir)
    a = reader.get_tensor('InceptionV1/Conv2d_1a_7x7/weights')
    max = np.amax(a)
    min = np.amin(a)
    # print a
    print 'max:', max
    print 'min:', min
    for i in a:
        i = 255 * (i - min) / (max - min)
    # a = np.clip(a,0,255)
    filters = np.split(a, 64, 3)
    for ii in range(64):
        filters[ii] = np.reshape(filters[ii], [7, 7, 3])
        imsave('./out/%d.bmp' % (ii), filters[ii], 'bmp')
    print(reader.get_tensor('InceptionV1/Conv2d_1a_7x7/weights').shape)

    BigImage = Image.new('RGBA', (8 * 7, 8 * 7))
    for y in xrange(8):
        for x in xrange(8):
            fname = './out/%d.bmp' % (8 * y + x)
            fromImage = Image.open(fname)
            BigImage.paste(fromImage, (x * 7, y * 7))
    BigImage.save('big_image.bmp', 'bmp')
    res = cv2.resize(np.array(BigImage), (32 * 7, 32 * 7), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('lena.bmp', res)


if __name__ == "__main__":
    main()
