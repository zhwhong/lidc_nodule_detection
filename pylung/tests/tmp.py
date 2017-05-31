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
            dstpath = os.path.join(save_file_path, 'original%d.jpg' % (dcm_num,))
            scipy.misc.imsave(dstpath, dcm.pixel_array)
            ct_image = Image.open(dstpath)

            num = 0
            for point in info[sop_uid]['nodules']:
                for i in range(len(tmp)):
                    box = (int(point['centroid'][0] + tmp[i][0]), int(point['centroid'][1] + tmp[i][1]),
                           int(point['centroid'][0] + tmp[i][2]), int(point['centroid'][1] + tmp[i][3]))
                    ct_image.crop(box).save(os.path.join(save_file_path, 'dcm_%d_split_%d.jpg' % (dcm_num, num)), 'jpeg')
                    writer.writerow(['dcm_%d_split_%d.jpg' % (dcm_num, num), 1])
                    num += 1
            for point in info[sop_uid]['small_nodules']:
                for i in range(len(tmp)):
                    box = (int(point['centroid'][0] + tmp[i][0]), int(point['centroid'][1] + tmp[i][1]),
                           int(point['centroid'][0] + tmp[i][2]), int(point['centroid'][1] + tmp[i][3]))
                    ct_image.crop(box).save(os.path.join(save_file_path, 'dcm_%d_split_%d.jpg' % (dcm_num, num)), 'jpeg')
                    writer.writerow(['dcm_%d_split_%d.jpg' % (dcm_num, num), 2])
                    num += 1
            for point in info[sop_uid]['non_nodules']:
                for i in range(len(tmp)):
                    box = (int(point['centroid'][0] + tmp[i][0]), int(point['centroid'][1] + tmp[i][1]),
                           int(point['centroid'][0] + tmp[i][2]), int(point['centroid'][1] + tmp[i][3]))
                    ct_image.crop(box).save(os.path.join(save_file_path, 'dcm_%d_split_%d.jpg' % (dcm_num, num)), 'jpeg')
                    writer.writerow(['dcm_%d_split_%d.jpg' % (dcm_num, num), 3])
                    num += 1
            dcm_num += 1