import numpy as N
import os
import cv2

import sys
import struct
import cv2
import gc
import array

try:
    import cPickle as pickle
except ImportError:
    import pickle





def get_volume(gs):
    return (gs[1]-gs[0] + 1)*(gs[4]-gs[2] + 1)*(gs[5]-gs[3] + 1)

def get_area(gs):
    return (gs[4]-gs[2] + 1)*(gs[5]-gs[3] + 1)

def combined_sample(gs1, gs2):
    if gs1 == [] or gs2 == []:
        return 0, []

    z1 = max(gs1[0], gs2[0])
    z2 = min(gs1[1], gs2[1])
    x1 = max(gs1[2], gs2[2])
    y1 = max(gs1[3], gs2[3])
    x2 = min(gs1[4], gs2[4])
    y2 = min(gs1[5], gs2[5])
    gs = []
    flag = 0
    if(z1 <= z2 and x1 <= x2 and y1 <= y2):
        v0 = get_volume([z1, z2, x1, y1, x2, y2])
        v1 = get_volume(gs1)
        v2 = get_volume(gs2)

        if(v0 > 0.5 * min(v1, v2)):
            z1 = min(gs1[0], gs2[0])
            z2 = max(gs1[1], gs2[1])
            x1 = min(gs1[2], gs2[2])
            y1 = min(gs1[3], gs2[3])
            x2 = max(gs1[4], gs2[4])
            y2 = max(gs1[5], gs2[5])

            gs = [z1, z2, x1, y1, x2, y2]
            flag = 1

    return flag, gs

def check_in2d(gs1, gs2):
    z1 = max(gs1[0], gs2[0])
    z2 = min(gs1[1], gs2[1])
    x1 = max(gs1[2], gs2[2])
    y1 = max(gs1[3], gs2[3])
    x2 = min(gs1[4], gs2[4])
    y2 = min(gs1[5], gs2[5])
    gs = []
    flag = 0
    if(z1 <= z2 and x1 <= x2 and y1 <= y2):
        v0 = get_area([z1, z2, x1, y1, x2, y2])
        v1 = get_area(gs1)
        v2 = get_area(gs2)

        if(v0 > 0.5 * min(v1, v2)):
            z1 = min(gs1[0], gs2[0])
            z2 = max(gs1[1], gs2[1])
            x1 = min(gs1[2], gs2[2])
            y1 = min(gs1[3], gs2[3])
            x2 = max(gs1[4], gs2[4])
            y2 = max(gs1[5], gs2[5])

            gs = [z1, z2, x1, y1, x2, y2]
            flag = 1

    return flag, gs

def get_single_matched_list(org_sample, det_sample):
    flag, _ = check_in2d(org_sample, det_sample)
    res_list = []
    if(flag == 1):
        for index in range(det_sample[0], det_sample[1] + 1):
            if index in range(org_sample[0], org_sample[1] + 1):
                res_list.append(det_sample[6][index - det_sample[0]])

    return res_list

def commpare_sample(org, det):
    matched_scores_list = list()
    gd_num = len(org)
    dt_num = len(det)
    for org_sample in org:
        for det_sample in det:
            l_s = get_single_matched_list(org_sample, det_sample)

            if(l_s != []):
                matched_scores_list.append(min(l_s))

    return matched_scores_list, gd_num, dt_num

def modify_rc(sample6):
    dis_x = sample6[4] - sample6[2]
    dis_y = sample6[5] - sample6[3]

    if dis_x < 32:
        dis_x = (32 - dis_x) / 2
        sample6[4] + dis_x
        sample6[2] - dis_x


    if dis_y < 32:
        dis_y = (32 - dis_y) / 2
        sample6[5] + dis_y
        sample6[3] - dis_y

    return sample6

def merge_gd(gd_sample_set):
    # for i1 in range(len(gd_sample_set)):
    #     gd_sample_set[i1] = modify_rc(gd_sample_set[i1])

    for i1 in range(len(gd_sample_set)):
        for i2 in range(i1 + 1, len(gd_sample_set)):
            f, merge_sample = combined_sample(gd_sample_set[i1], gd_sample_set[i2])
            if(f == 1):
                gd_sample_set[i1] = merge_sample
                gd_sample_set[i2] = []
    while [] in gd_sample_set:
        gd_sample_set.remove([])
    return gd_sample_set

def get_dt_scores(sample_dt):
    sc_list = list()
    for i in range(len(sample_dt)):
        sc = list()
        for j in range(len(sample_dt[i][6])):
            sc.append(sample_dt[i][6][j])
        sc_list.append(min(sc))
    return sc_list

def main():
    scores_set = set([])

    # f = open('dump_dt.txt', 'rb')
    # compare_info_dt= pickle.load(f)
    # f.close()
    #
    # f = open('dump_gd.txt', 'rb')
    # compare_info_gd= pickle.load(f)
    # f.close()
    #
    # compare_info = []
    # for dt in compare_info_dt:
    #     for gd in compare_info_gd:
    #         if (dt[0] == gd[0]):
    #             compare_info.append([dt[0], gd[1], dt[2]])

    # f = open('dump.txt', 'wb')
    # pickle.dump(compare_info, f)
    # f.close()

    f = open('dump.txt', 'rb')
    compare_info = pickle.load(f)
    f.close()


    list()
    for sample in compare_info:
        sample[1] = merge_gd(sample[1])
        for dets in sample[2]:
            for score in dets[6]:
                scores_set.add(score)

    scores_all_list = list(scores_set)
    scores_match_list = list(scores_set)

    scores_all_list.sort(cmp=lambda x, y: cmp(x, y))
    scores_match_list.sort(cmp=lambda x, y: cmp(x, y))

    score2index = dict()
    for n in range(len(scores_match_list)):
        score2index[scores_match_list[n]] = n
        scores_match_list[n] = [scores_match_list[n], 0]
        scores_all_list[n] = [scores_all_list[n], 0]
        
    all_gd_num = 0
    all_dt_num = 0

    for sample in compare_info:
        matched_score_list, gd_num, dt_num = commpare_sample(sample[1], sample[2])

        all_gd_num = all_gd_num + gd_num
        all_dt_num = all_dt_num + dt_num

        for s in matched_score_list:
            scores_match_list[score2index[s]][1] = scores_match_list[score2index[s]][1] + 1

        single_dt_sc = get_dt_scores(sample[2])
        for s in single_dt_sc:
            scores_all_list[score2index[s]][1] = scores_all_list[score2index[s]][1] + 1
        #scores_all_list[score2index[single_dt_sc]][1] = scores_all_list[score2index[single_dt_sc]][1] + 1

        nnn = []
        nnn.extend(matched_score_list)
        nnn.extend(single_dt_sc)

        #if(scores_all_list[3927][0] in nnn):
        #    print scores_all_list[3927],  scores_match_list[3927]

    c1 = []
    c2 = []

    l = (range(len(scores_match_list) - 1))
    l.reverse()

    for i in l:
        scores_match_list[i][1] = scores_match_list[i][1] + scores_match_list[i + 1][1]


    l = (range(len(scores_all_list) - 1))
    l.reverse()
    for i in l:
        scores_all_list[i][1] = scores_all_list[i][1] + scores_all_list[i + 1][1]

    score = list()
    dt_num_list = list()
    match_num_list = list()

    for index in range(len(scores_match_list)):
        score.append(scores_match_list[index][0])
        dt_num_list.append(scores_all_list[index][1])
        match_num_list.append(scores_match_list[index][1])



    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.figure(2)

    score = N.array(score)
    dt_num_list = N.array(dt_num_list)
    #dt_num_list = 1 - dt_num_list / (1.* all_dt_num)


    match_num_list = N.array(match_num_list)
    #match_num_list = 1.* match_num_list / all_gd_num

    index = 0
    for i in range(len(dt_num_list)):
        if(dt_num_list[i] == 0):
            index = i
            break


    dt_num_list = dt_num_list[:index]
    match_num_list = match_num_list[:index]
    score = score[:index]

    for i in range(len(dt_num_list)):
        if (dt_num_list[i] < match_num_list[i]):
            dt_num_list[i] = match_num_list[i]


    pres_l = 1.*match_num_list/dt_num_list




    recall_l = 1.* match_num_list/all_gd_num



    plt.figure(1)
    plt.plot(score, pres_l)
    plt.plot(score, recall_l)

    plt.figure(2)
    plt.plot(pres_l, recall_l)

    plt.show()
    plt.waitforbuttonpress()


#--gpu=0
if __name__ == '__main__':
    main()
