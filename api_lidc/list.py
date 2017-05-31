import os
import argparse
import dicom
from annotation2 import find_all_files

DATA_PATH = './LIDC-IDRI-0001/'
OUT_PATH = './output/'


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--srcpath', type=str, default=DATA_PATH, help='the dcm image path')
    parser.add_argument('-d', '--dstpath', type=str, default=OUT_PATH, help='the output path')
    return parser


def analyse(srcpath, dstpath):
    for f_dcm in find_all_files(srcpath, suffix='.dcm'):
        dcm = dicom.read_file(f_dcm)
        company = dcm.get('Manufacturer')
        if company != 'GE MEDICAL SYSTEMS':
            return
        else:
            print srcpath
            break


def main():
    parser = build_parser()
    options = parser.parse_args()

    assert os.path.exists(options.srcpath)

    if not os.path.exists(options.dstpath):
        os.mkdir(options.dstpath)

    # print 'DATA_PATH: ', options.srcpath
    # print 'OUT_PATH: ', options.dstpath

    analyse(options.srcpath, options.dstpath)


if __name__ == '__main__':
    main()
