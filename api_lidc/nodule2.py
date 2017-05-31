import os
import argparse
import dicom
import pickle
from annotation2 import parse_dir
from annotation2 import find_all_files

DATA_PATH = './LIDC-IDRI-0001/'
OUT_PATH = '/baina/sda1/data/lidc_matrix/TXT/'


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--srcpath', type=str, default=DATA_PATH, help='the dcm image path')
    parser.add_argument('-d', '--dstpath', type=str, default=OUT_PATH, help='the output path')
    return parser


def annotation(srcpath, dstpath):
    parse_dir(srcpath, dstpath)
    f = open(os.path.join(dstpath, 'annotation_flatten.pkl'))
    info = pickle.load(f)
    f.close()

    slicers = 0
    rows = 512
    cols = 512
    data_type = 4
    slice_thickness = 0
    pixel_spacing_x = 0
    pixel_spacing_y = 0
    flag = False

    instance_list = {}
    for f_dcm in find_all_files(srcpath, suffix='.dcm'):
        slicers += 1
        dcm = dicom.read_file(f_dcm)
        if not flag:
            flag = True
            rows = dcm[0x0028, 0x0010].value
            cols = dcm[0x0028, 0x0011].value
            slice_thickness = dcm[0x0018, 0x0050].value
            pixel_spacing_x, pixel_spacing_y = dcm[0x0028, 0x0030].value

        sop_uid = dcm[0x08, 0x18].value
        # z = dcm.get('SliceLocation')
        z = dcm[0x0020, 0x0013].value  # instance number
        instance_list[z] = sop_uid

    # sort by z position
    instances = []
    for i in sorted(instance_list.keys(), reverse=True):
        instances.append(instance_list[i])
    map_uid = {}
    for i in range(len(instances)):
        map_uid[instances[i]] = i

    # name = srcpath.split('/')[0]
    name = srcpath.split('/')[6]

    f2 = open(os.path.join(dstpath, '%s.dat.txt' % (name,)), 'w+')

    w = []
    w.append(str(slicers) + ' ')
    w.append(str(rows) + ' ')
    w.append(str(cols) + ' ')
    w.append(str(data_type) + ' ')
    w.append(str(pixel_spacing_x)+' ')
    w.append(str(pixel_spacing_y)+' ')
    w.append(str(slice_thickness)+'\n')
    f2.writelines(w)

    for nodule in info['nodules']:
        w = []
        w.append('1 ')
        length = 0
        for point in nodule:
            length += len(point['pixels'])
        w.append(str(3 * length) + ' ')
        for point in nodule:
            for i in point['pixels']:
                w.append(str(i[0]) + ' ')
                w.append(str(i[1]) + ' ')
                w.append(str(map_uid[point['sop_uid']]) + ' ')
        w.append('\n')
        f2.writelines(w)

    for nodule in info['small_nodules']:
        w = []
        w.append('2 ')
        length = 0
        for point in nodule:
            length += len(point['pixels'])
        w.append(str(3 * length) + ' ')
        for point in nodule:
            for i in point['pixels']:
                w.append(str(i[0]) + ' ')
                w.append(str(i[1]) + ' ')
                w.append(str(map_uid[point['sop_uid']]) + ' ')
        w.append('\n')
        f2.writelines(w)

    for nodule in info['non_nodules']:
        w = []
        w.append('3 ')
        length = 0
        for point in nodule:
            length += len(point['pixels'])
        w.append(str(3 * length) + ' ')
        for point in nodule:
            for i in point['pixels']:
                w.append(str(i[0]) + ' ')
                w.append(str(i[1]) + ' ')
                w.append(str(map_uid[point['sop_uid']]) + ' ')
        w.append('\n')
        f2.writelines(w)

    f2.close()


def main():
    parser = build_parser()
    options = parser.parse_args()

    assert os.path.exists(options.srcpath)

    if not os.path.exists(options.dstpath):
        os.mkdir(options.dstpath)

    print 'DATA_PATH: ', options.srcpath
    print 'OUT_PATH: ', options.dstpath

    annotation(options.srcpath, options.dstpath)


if __name__ == '__main__':
    main()
