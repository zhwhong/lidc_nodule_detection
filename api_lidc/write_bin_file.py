import os
import argparse
import dicom
import numpy as np
import struct

DATA_PATH = './LIDC-IDRI-0001/'
OUT_PATH = './output/'

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--srcpath', type=str, default=DATA_PATH, help='the dcm image path')
    parser.add_argument('-d', '--dstpath', type=str, default=OUT_PATH, help='the output path')
    return parser

def find_all_files(root, suffix=None):
    res = []
    for root, _, files in os.walk(root):
        for f in files:
            if suffix is not None and not f.endswith(suffix):
                continue
            res.append(os.path.join(root, f))
    return res


def main():
    parser = build_parser()
    options = parser.parse_args()

    assert os.path.exists(options.srcpath)

    if not os.path.exists(options.dstpath):
        os.mkdir(options.dstpath)

    print 'DATA_PATH: ', options.srcpath
    print 'OUT_PATH: ', options.dstpath

    name = options.srcpath.split('/')[6]

    image_list = {}
    for f_dcm in find_all_files(options.srcpath, suffix='.dcm'):
        # print 'reading dicom image %s ...' % (f_dcm[-4 - 6:])
        dcm = dicom.read_file(f_dcm)
        # z = dcm.get('SliceLocation') # some patient examples may have problem in this way
        z = dcm[0x0020, 0x0013].value  # instance number
        image_list[z] = np.array(dcm.pixel_array)

    # sort by z position
    images = []
    for i in sorted(image_list.keys(), reverse=True):
    # for i in sorted(image_list.keys()):
        images.append(image_list[i])

    images = np.array(images)
    print 'Matrix shape: ', images.shape
    h, w, l = images.shape
    img = images.reshape(h, w*l)
    file = open(os.path.join(options.dstpath, '%s.dat' % (name,)), "wb")
    for i in range(h):
        print 'writing image %d / %d ...' % (i, h-1)
        file.write(struct.pack("%dh"%(w*l,), *img[i]))
    file.close()


if __name__ == '__main__':
    main()
