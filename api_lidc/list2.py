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
    f = open(os.path.join('./', 'out.txt'), 'a+')
    w = []
    name = srcpath.split('/')[6]
    w.append(name+' ')
    for f_dcm in find_all_files(srcpath, suffix='.dcm'):
        dcm = dicom.read_file(f_dcm)
        company = dcm.get('Manufacturer')
        slice_thickness = dcm[0x0018, 0x0050].value
        z = dcm.get('SliceLocation')
        pixel_spacing = dcm[0x0028, 0x0030].value
        w.append(company+' ')
        w.append(str(slice_thickness)+' ')
        w.append(str(pixel_spacing[0])+' ')
        w.append(str(pixel_spacing[1])+' ')
        w.append(str(z) + '\n')
        break
    print w
    f.writelines(w)
    f.close()


def main():
    parser = build_parser()
    options = parser.parse_args()

    assert os.path.exists(options.srcpath)

    if not os.path.exists(options.dstpath):
        os.mkdir(options.dstpath)

    print 'DATA_PATH: ', options.srcpath
    print 'OUT_PATH: ', options.dstpath

    analyse(options.srcpath, options.dstpath)


if __name__ == '__main__':
    main()
