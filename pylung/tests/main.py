#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
import argparse
import pickle
import csv
import dicom
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
#import image2gif

curdir = os.path.dirname(__file__)
sys.path.append(os.path.join(curdir, '../..'))

from pylung.annotation import parse_dir
from pylung.detector import parenchyma_block
from pylung.utils import find_all_files
from pylung.filters.parenchyma import _get_parenchyma_mask

DATA_PATH = '../data/LIDC-IDRI-0001/'
# DATA_PATH = '../data/testdir/'
OUT_PATH = './output/'


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type = str, default = 'view', help='the mode(data cut or view ?)')
    parser.add_argument('-m', '--method', type=str, default='original', help='the cut or view method')
    parser.add_argument('-s', '--srcpath', type=str, default=DATA_PATH, help='the dcm image path')
    parser.add_argument('-d', '--dstpath', type=str, default=OUT_PATH, help='the output path')
    return parser


def is_in(centroid, field):
    flag = 1 if (field[2] >= centroid[0] >= field[0] and field[3] >= centroid[1] >= field[1]) else 0
    return flag


def parenchyma_cut(dcm_file_path, save_file_path):
    '''
    class: without annotation:0, nodule:1, small_nodule:2, non_nodule:3
    '''
    f = open(os.path.join(dcm_file_path, 'annotation_flatten.pkl'))
    info = pickle.load(f)
    f.close()
    csvfile = file(os.path.join(save_file_path, 'split_parenchyma_info.csv'), 'wb')
    writer = csv.writer(csvfile)

    dcm_num = 0
    for f_dcm in find_all_files(dcm_file_path, '.dcm'):
        print 'processing dicom image %s ...' % (f_dcm[-4-6:])
        dcm = dicom.read_file(f_dcm)
        sop_uid = dcm[0x08, 0x18].value
        if info.has_key(sop_uid):
            #dir_name = os.path.join(save_file_path, 'dcm_split%d' % (dcm_num,))
            #os.mkdir(dir_name)
            dir_name = save_file_path
            dstpath = os.path.join('./tmp/', 'original%d.jpg' % (dcm_num,))
            scipy.misc.imsave(dstpath,dcm.pixel_array)
            ct_image = Image.open(dstpath)

            num = 0
            for i in parenchyma_block(f_dcm):
                box = (i[0], i[1], i[0]+i[2], i[1]+i[3])
                ct_image.crop(box).save(os.path.join(dir_name, 'dcm_%d_split_%d_p.jpg' % (dcm_num, num)), 'jpeg')
                flag = 0
                for point in info[sop_uid]['nodules']:
                    if is_in(point['centroid'], box):
                        flag = 1
                        writer.writerow(['dcm_%d_split_%d_p.jpg' % (dcm_num, num), 1])
                for point in info[sop_uid]['small_nodules']:
                    if is_in(point['centroid'], box):
                        flag = 1
                        writer.writerow(['dcm_%d_split_%d_p.jpg' % (dcm_num, num), 2])
                for point in info[sop_uid]['non_nodules']:
                    if is_in(point['centroid'], box):
                        flag = 1
                        writer.writerow(['dcm_%d_split_%d_p.jpg' % (dcm_num, num), 3])
                if flag == 0:
                    writer.writerow(['dcm_%d_split_%d_p.jpg' % (dcm_num, num), 0])
                num += 1
            dcm_num += 1


def annotation_cut(dcm_file_path, save_file_path):
    '''
    :param dcm_file_path:
    :param save_file_path:
    :return:
    class: without annotation:0, nodule:1, small_nodule:2, non_nodule:3
    '''
    f = open(os.path.join(dcm_file_path, 'annotation_flatten.pkl'))
    info = pickle.load(f)
    f.close()
    csvfile = file(os.path.join(save_file_path, 'split_with_annotation_info.csv'), 'wb')
    writer = csv.writer(csvfile)
    dcm_num = 0
    for f_dcm in find_all_files(dcm_file_path, '.dcm'):
        print 'processing dicom image %s ...' % (f_dcm[-4 - 6:]),f_dcm
        dcm = dicom.read_file(f_dcm)
        sop_uid = dcm[0x08, 0x18].value
        if info.has_key(sop_uid):
            dstpath = os.path.join('./tmp/', 'original%d.jpg' % (dcm_num,))
            scipy.misc.imsave(dstpath, dcm.pixel_array)
            ct_image = Image.open(dstpath)

            num = 0
            for point in info[sop_uid]['nodules']:
                box = (int(point['centroid'][0] - 32), int(point['centroid'][1] - 32), int(point['centroid'][0] + 32),
                       int(point['centroid'][1] + 32))
                ct_image.crop(box).save(os.path.join(save_file_path, 'dcm_%d_split_%d_a.jpg' % (dcm_num, num)), 'jpeg')
                writer.writerow(['dcm_%d_split_%d_a.jpg' % (dcm_num, num), 1])
                num += 1
            for point in info[sop_uid]['small_nodules']:
                box = (int(point['centroid'][0] - 32), int(point['centroid'][1] - 32), int(point['centroid'][0] + 32),
                       int(point['centroid'][1] + 32))
                ct_image.crop(box).save(os.path.join(save_file_path, 'dcm_%d_split_%d_a.jpg' % (dcm_num, num)), 'jpeg')
                writer.writerow(['dcm_%d_split_%d_a.jpg' % (dcm_num, num), 2])
                num += 1
            for point in info[sop_uid]['non_nodules']:
                box = (int(point['centroid'][0] - 32), int(point['centroid'][1] - 32), int(point['centroid'][0] + 32),
                       int(point['centroid'][1] + 32))
                ct_image.crop(box).save(os.path.join(save_file_path, 'dcm_%d_split_%d_a.jpg' % (dcm_num, num)), 'jpeg')
                writer.writerow(['dcm_%d_split_%d_a.jpg' % (dcm_num, num), 3])
                num += 1
            dcm_num += 1


def annotation_cut2(dcm_file_path, save_file_path):
    '''
    :param dcm_file_path:
    :param save_file_path:
    :return:
    class: without annotation:0, nodule:1, small_nodule:2, non_nodule:3
    '''
    f = open(os.path.join(dcm_file_path, 'annotation_flatten.pkl'))
    info = pickle.load(f)
    f.close()
    csvfile = file(os.path.join(save_file_path, 'split_with_annotation_info.csv'), 'wb')
    writer = csv.writer(csvfile)
    dcm_num = 0
    tmp = [[-32, -32, 32, 32],
           [-48, -32, 16, 32],
           [-16, -32, 48, 32],
           [-32, -48, 32, 16],
           [-32, -16, 32, 48],
           [-48, -48, 16, 16],
           [-16, -16, 48, 48],
           [-48, -16, 16, 48],
           [-16, -48, 48, 16]
          ]
    for f_dcm in find_all_files(dcm_file_path, '.dcm'):
        print 'processing dicom image %s ...' % (f_dcm[-4 - 6:])
        dcm = dicom.read_file(f_dcm)
        sop_uid = dcm[0x08, 0x18].value
        if info.has_key(sop_uid):
            dstpath = os.path.join('./tmp/', 'original%d.jpg' % (dcm_num,))
            scipy.misc.imsave(dstpath, dcm.pixel_array)
            ct_image = Image.open(dstpath)

            num = 0
            for point in info[sop_uid]['nodules']:
                for i in range(len(tmp)):
                    box = (int(point['centroid'][0] + tmp[i][0]), int(point['centroid'][1] + tmp[i][1]),
                           int(point['centroid'][0] + tmp[i][2]), int(point['centroid'][1] + tmp[i][3]))
                    ct_image.crop(box).save(os.path.join(save_file_path, 'dcm_%d_split_%d_a.jpg' % (dcm_num, num)), 'jpeg')
                    writer.writerow(['dcm_%d_split_%d_a.jpg' % (dcm_num, num), 1])
                    num += 1
            for point in info[sop_uid]['small_nodules']:
                for i in range(len(tmp)):
                    box = (int(point['centroid'][0] + tmp[i][0]), int(point['centroid'][1] + tmp[i][1]),
                           int(point['centroid'][0] + tmp[i][2]), int(point['centroid'][1] + tmp[i][3]))
                    ct_image.crop(box).save(os.path.join(save_file_path, 'dcm_%d_split_%d_a.jpg' % (dcm_num, num)), 'jpeg')
                    writer.writerow(['dcm_%d_split_%d_a.jpg' % (dcm_num, num), 2])
                    num += 1
            for point in info[sop_uid]['non_nodules']:
                for i in range(len(tmp)):
                    box = (int(point['centroid'][0] + tmp[i][0]), int(point['centroid'][1] + tmp[i][1]),
                           int(point['centroid'][0] + tmp[i][2]), int(point['centroid'][1] + tmp[i][3]))
                    ct_image.crop(box).save(os.path.join(save_file_path, 'dcm_%d_split_%d_a.jpg' % (dcm_num, num)), 'jpeg')
                    writer.writerow(['dcm_%d_split_%d_a.jpg' % (dcm_num, num), 3])
                    num += 1
            dcm_num += 1

def original_view(srcpath, dstpath):
    image_list = {}
    for f_dcm in find_all_files(srcpath, suffix='.dcm'):
        print 'processing dicom image %s ...' % (f_dcm[-4 - 6:])
        name = f_dcm[-4 - 6:-4]
        dcm = dicom.read_file(f_dcm)
        z = dcm.get('SliceLocation')
        scipy.misc.imsave(os.path.join(dstpath, 'original_%s.jpg' % (name,)), dcm.pixel_array)
        image_list[z] = (Image.open(os.path.join(dstpath, 'original_%s.jpg' % (name,))))

    images = []
    for i in sorted(image_list.keys(), reverse=True):
        images.append(image_list[i])
    print len(image_list)
    #image2gif.writeGif(os.path.join(dstpath,'original.gif'), images, duration=0.3, nq=0.5)

def original_view2(srcpath, dstpath):
    image_list = {}
    for f_dcm in find_all_files(srcpath, suffix='.dcm'):
        print 'processing dicom image %s ...' % (f_dcm[-4 - 6:])
        name = f_dcm[-4 - 6:-4]
        dcm = dicom.read_file(f_dcm)
        z = dcm.get('SliceLocation')
        image_list[z] =np.array(dcm.pixel_array)

    images = []
    for i in sorted(image_list.keys(), reverse=True):
        images.append(image_list[i])

    images = np.array(images)

    l,  w, h  = images.shape
    print (l,w,h)
    for i in range(w):
        tmp = images[:, i, :]
        path = os.path.join(dstpath, 'tmp_%s.jpg' % (i,))
        outpath =path
        scipy.misc.imsave(path, tmp)
        a = Image.open(path)
        b = a.resize((512,int(133*2.5/0.703125)))
        b.save(outpath)




def parenchyma_view(srcpath, dstpath):
    for f_dcm in find_all_files(srcpath, suffix='.dcm'):
        print 'processing dicom image %s ...' % (f_dcm[-4 - 6:])
        name = f_dcm[-4-6:-4]
        dcm = dicom.read_file(f_dcm)
        image = np.array(dcm.pixel_array)
        result = _get_parenchyma_mask(image, 625)
        scipy.misc.imsave(os.path.join(dstpath, 'parenchyma_%s.jpg' % (name,)), result)
        w, h = result.shape[:2]
        img2 = np.zeros([w, h])
        for i in range(w):
            for j in range(h):
                if result[i][j] == 255:
                    img2[i][j] = image[i][j]
        scipy.misc.imsave(os.path.join(dstpath, 'parenchyma2_%s.jpg' % (name,)), img2)


def annotation_view(srcpath, dstpath):
    f = open(os.path.join(srcpath, 'annotation_flatten.pkl'))
    info = pickle.load(f)
    f.close()
    j = 0
    image_list = {}
    for f_dcm in find_all_files(srcpath, suffix='.dcm'):
        print 'processing dicom image %s ...' % (f_dcm[-4 - 6:])
        name = f_dcm[-4 - 6:-4]
        dcm = dicom.read_file(f_dcm)
        sop_uid = dcm[0x08, 0x18].value
        z = dcm.get('SliceLocation')
        if info.has_key(sop_uid):
            img = dcm.pixel_array
            extent = (0, 512, 0, 512)
            plt.imshow(img, cmap=plt.cm.gray, origin='upper', extent=extent)
            print j, ' figures:'

            for point in info[sop_uid]['nodules']:
                x1 = point['field'][0]
                y1 = 512 - point['field'][1]
                x2 = point['field'][0]
                y2 = 512 - point['field'][3]
                x3 = point['field'][2]
                y3 = 512 - point['field'][3]
                x4 = point['field'][2]
                y4 = 512 - point['field'][1]
                plt.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], 'r-', linewidth=2, alpha=1)
            for point in info[sop_uid]['small_nodules']:
                x = point['centroid'][0]
                y = 512 - point['centroid'][1]
                print 'small_nodules', x, y
                plt.plot(x, y, 'bv', alpha=0.5)
            for point in info[sop_uid]['non_nodules']:
                x = point['centroid'][0]
                y = 512 - point['centroid'][1]
                print 'non_nodules:', x, y
                plt.plot(x, y, 'gx', alpha=1)

            plt.savefig(os.path.join(dstpath, '%s.jpg' % (name,)))
            image_list[z] = (Image.open(os.path.join(dstpath, '%s.jpg' % (name,))))
            plt.close('all')
            j += 1
    images = []
    for i in sorted(image_list.keys(), reverse=True):
        images.append(image_list[i])
    #image2gif.writeGif(os.path.join(dstpath, 'annotated.gif'), images, duration=0.3, nq=0.5)


def main():
    parser = build_parser()
    options = parser.parse_args()

    assert os.path.exists(options.srcpath)

    if not os.path.exists(options.dstpath):
        os.mkdir(options.dstpath)

    print 'DATA_PATH: ', options.srcpath
    print 'OUT_PATH: ', options.dstpath

    flag = True

    # multiply test cases
    for file in os.listdir(options.srcpath):
        path = os.path.join(options.srcpath, file)
        if os.path.isdir(path) and find_all_files(path, '.dcm') is not None:
            print "multiply test cases"
            flag = False
            dstpath = os.path.join(options.dstpath, file)
            if os.path.exists(dstpath):
                pass
                #os.removedirs(dstpath)
            else:
                os.mkdir(dstpath)
            parse_dir(path)
            if options.mode == 'view':
                if options.method == 'original':
                    original_view(path, dstpath)
                elif options.method == 'parenchyma':
                    parenchyma_view(path, dstpath)
                elif options.method == 'annotation':
                    annotation_view(path, dstpath)
                else:
                    print 'wrong method !!!'
            elif options.mode == 'cut':
                if options.method == 'parenchyma':
                    parenchyma_cut(path, dstpath)
                elif options.method == 'annotation':
                    annotation_cut(path, dstpath)
                else:
                    print 'wrong method !!!'

    # single test case
    if flag and find_all_files(options.srcpath, '.xml') and find_all_files(options.srcpath, '.dcm'):
        print "single test case"
        parse_dir(options.srcpath)
        if options.mode == 'view':
            if options.method == 'original':
                original_view(options.srcpath, options.dstpath)
            elif options.method == 'parenchyma':
                parenchyma_view(options.srcpath, options.dstpath)
            elif options.method == 'annotation':
                annotation_view(options.srcpath, options.dstpath)
            else:
                print 'wrong method !!!'
        elif options.mode == 'cut':
            if options.method == 'parenchyma':
                parenchyma_cut(options.srcpath, options.dstpath)
            elif options.method == 'annotation':
                annotation_cut(options.srcpath, options.dstpath)
            else:
                print 'wrong method !!!'


if __name__ == '__main__':
    main()
