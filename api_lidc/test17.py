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
    f = open(os.path.join(dstpath, 'out.txt'), 'a+')
    for f_dcm in find_all_files(srcpath, suffix='.dcm'):
        w = []
        dcm = dicom.read_file(f_dcm)
        sop_uid = dcm[0x08, 0x18].value
        company = dcm.get('Manufacturer')
        slice_thickness = dcm[0x0018, 0x0050].value
        image_position = dcm[0x0020, 0x0032].value
        z = dcm.get('SliceLocation')
        ins = dcm[0x0020, 0x0013].value
        pixel_spacing = dcm[0x0028, 0x0030].value
        w.append(str(ins) + ' ')
        w.append(company+' ')
        w.append(sop_uid+' ')
        w.append(str(slice_thickness)+' ')
        w.append('(' + str(pixel_spacing[0])+' ')
        w.append(str(pixel_spacing[1])+') ')
        w.append('(' + str(image_position[0])+' ')
        w.append(str(image_position[1]) + ' ')
        w.append(str(image_position[2]) + ') ')
        w.append('z: '+str(z) + '\n')
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
