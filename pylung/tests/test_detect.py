import os
import sys
import numpy as np
import dicom
import scipy.misc
import matplotlib.pyplot as plt

curdir = os.path.dirname(__file__)
sys.path.append(os.path.join(curdir, '../..'))

import unittest

from pylung.detector import NoduleDetector
from pylung.filters.parenchyma import _get_parenchyma_mask


class NoduleDetectorTest(unittest.TestCase):
    def test_1(self):
        print '===========test1================='
        curdir = os.path.dirname(__file__)
        dcm = os.path.join(curdir, 'ct.dcm')
        dcm = dicom.read_file(dcm)
        image = np.array(dcm.pixel_array)
        scipy.misc.imsave(os.path.join(curdir, 'original.png'), image)
        mask = _get_parenchyma_mask(image, 625)
        scipy.misc.imsave(os.path.join(curdir, 'parenchyma.png'), mask)
        h, w = mask.shape[:2]
        img2 = np.zeros([h,w])
        for i in range(h):
            for j in range(w):
                if mask[i][j] == 255:
                    img2[i][j] = image[i][j]
        scipy.misc.imsave(os.path.join(curdir, 'parenchyma2.png'), img2)
        img3 = np.zeros([h, w])
        for i in range(h):
            for j in range(w / 2):
                if mask[i][j] == 255:
                    img3[i][j] = image[i][j]
        scipy.misc.imsave(os.path.join(curdir, 'parenchyma_half.png'), img3)
        img4 = np.zeros([h, w])
        for i in range(h):
            for j in range(w):
                if j < w/2:
                    img4[i][j] = image[i][j]
                else:
                    img4[i][j] = 0
        scipy.misc.imsave(os.path.join(curdir, 'original_half.png'), img4)

        """
        post_filter = []
        for candidate in NoduleDetector(os.path.join(curdir, 'ct.dcm')).candidates:
            centroid = [candidate[0] + candidate[2] / 2, candidate[1] + candidate[3] / 2]
            if mask[centroid[0]][centroid[1]] > 0:
                post_filter.append(candidate)

        print 'in parenchyma: ', len(post_filter)

        extent = (0, 512, 0, 512)
        plt.imshow(image, cmap=plt.cm.gray, origin='upper', extent=extent)

        for candidate in post_filter:
            x1 = candidate[1]
            y1 = 512 - candidate[0]
            x2 = x1
            y2 = 512 - candidate[0] - candidate[2]
            x3 = x1 + candidate[3]
            y3 = y2
            x4 = x3
            y4 = y1
            plt.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], 'b-', linewidth=3, alpha=0.8)

        plt.plot([0, 64, 64, 0, 0], [0, 0, 64, 64, 0], 'r-', linewidth=2, alpha=1)

        plt.savefig('test1_1.png')
        # plt.close()

        post_filter = []
        for candidate in NoduleDetector(os.path.join(curdir, 'ct.dcm')).candidates:
            centroid = [candidate[0] + candidate[2] / 2, candidate[1] + candidate[3] / 2]
            if mask[centroid[0]][centroid[1]] == 0:
                post_filter.append(candidate)

        print 'not in parenchyma: ', len(post_filter)

        # extent = (0, 512, 0, 512)
        # plt.imshow(image, cmap=plt.cm.gray, origin='upper', extent=extent)

        for candidate in post_filter:
            x1 = candidate[0]
            y1 = 512 - candidate[1]
            x2 = x1
            y2 = 512 - candidate[1] - candidate[3]
            x3 = x1 + candidate[2]
            y3 = y2
            x4 = x3
            y4 = y1
            plt.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], 'y-', linewidth=1, alpha=0.5)

        plt.savefig('test1_2.png')
        plt.close()
        """

    def test_2(self):
        print '\n===========test2================='
        curdir = os.path.dirname(__file__)
        dcm = os.path.join(curdir, 'ct.dcm')
        info = dicom.read_file('ct.dcm').pixel_array
        detector = NoduleDetector(dcm)
        result = detector.candidates
        print len(result)
        print result

        extent = (0, 512, 0, 512)
        plt.imshow(info, cmap=plt.cm.gray, origin='upper', extent=extent)

        for candidate in result:
            x1 = candidate[1]
            y1 = 512 - candidate[0]
            x2 = x1
            y2 = 512 - candidate[0] - candidate[2]
            x3 = x1 + candidate[3]
            y3 = y2
            x4 = x3
            y4 = y1
            plt.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], 'g:', linewidth=1, alpha=0.5)

        plt.savefig('test2.png')


    def test_3(self):
        print '\n===========test3================='
        curdir = os.path.dirname(__file__)
        dcm = os.path.join(curdir, 'ct.dcm')
        info = dicom.read_file('ct.dcm').pixel_array
        detector = NoduleDetector(dcm)
        result = detector.get_parenchyma_block()
        print len(result)
        print result

        extent = (0, 512, 0, 512)
        plt.imshow(info, cmap=plt.cm.gray, origin='upper', extent=extent)

        for candidate in result:
            x1 = candidate[1]
            y1 = 512 - candidate[0]
            x2 = x1
            y2 = 512 - candidate[0] - candidate[2]
            x3 = x1 + candidate[3]
            y3 = y2
            x4 = x3
            y4 = y1
            plt.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], 'b--', linewidth=1, alpha=1)

        plt.savefig('test3.png')



    def test_4(self):
        print '\n===========test4================='
        curdir = os.path.dirname(__file__)
        dcm = os.path.join(curdir, 'ct.dcm')
        info = dicom.read_file('ct.dcm').pixel_array
        # scipy.misc.imsave('ct.png', info)
        detector = NoduleDetector(dcm)
        result = detector.detect()
        print '================================'
        print len(result)
        print result

        extent = (0, 512, 0, 512)
        plt.imshow(info, cmap=plt.cm.gray, origin='upper', extent=extent)

        for candidate in result:
            x1 = candidate[1]
            y1 = 512 - candidate[0]
            x2 = x1
            y2 = 512 - candidate[0] - candidate[2]
            x3 = x1 + candidate[3]
            y3 = y2
            x4 = x3
            y4 = y1
            plt.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], 'r-', linewidth=2, alpha=1)

        plt.savefig('test4.png')

        # self.assertEqual(43, len(result))
        # self.assertTrue(len(result) == 43)


if __name__ == "__main__":
    unittest.main()