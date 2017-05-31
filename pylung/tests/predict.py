import sys
import os
import numpy as np
import csv
from scipy.misc import imread,imsave

curdir = os.path.dirname(__file__)
sys.path.append(os.path.join(curdir, '../..'))

from pylung.utils import gray_to_rgb,dcm_to_gray
from pylung.utils import dense_to_one_hot
from pylung.models.vgg_classifier import model


def train():
    model.load('pylung2.model')
    #return 0
    # start test
    csvfile = file('a.csv', 'rb')
    reader = csv.reader(csvfile)

    X, Y = [], []
    count = 0
    for line in reader:
        # imsave('./tmp/test_%s_%s' % (line[2], line[1]), gray_to_rgb(dcm_to_gray(imread(line[0] + line[1]))))
        X.append(gray_to_rgb(dcm_to_gray(imread(line[0] + line[1]))))
        Y.append([int(line[2])])
        # if count >= 50:
        #     break
        count += 1

    Y2=Y
    Y = dense_to_one_hot(np.array(Y), 2)

    #test
    out = model.predict(X)
    print out
    Z = [] #result list
    for i in out:
        Z.append(np.argmax(i))
    cnt = [0,0,0,0]
    for i in range(len(X)):
        if(Y2[i]==Z[i] and Y2[i][0]==1):
            cnt[0]+=1  #correct nodule(nodule)
            imsave('./tmp/TP_%d.png'%i, X[i])
        elif(Y2[i]==Z[i] and Y2[i][0]==0):
            #print Y2[i]
            cnt[1]+=1  #correct else(else)
            imsave('./tmp/TN_%d.png'%i, X[i])
        if(Y2[i]!=Z[i] and Y2[i][0]==1):
            cnt[2]+=1  #wrong nodule(else)
            print "wrong!!!!nodule(else) no.%d"%i
            imsave('./tmp/wrong_FN_%d.png'%i, X[i])
        if(Y2[i]!=Z[i] and Y2[i][0]==0):
            cnt[3]+=1  #wrong else(nodule)
            print "wrong!!!!else(nodule) no.%d"%i
            imsave('./tmp/wrong_FP_%d.png'%i, X[i])


    print cnt

    print "total accuracy : %f" % ((cnt[0] + cnt[1]) / float(cnt[0] + cnt[1] + cnt[2] + cnt[3]))
    print "nodule(nodule): %d, nodule(else): %d  accuracy: %f" % (cnt[0],cnt[2], float(cnt[0])/(cnt[0]+cnt[2]))
    print "else(else): %d, else(nodule): %d, accuracy: %f" % (cnt[1], cnt[3], float(cnt[1])/(cnt[1]+cnt[3]))


if __name__ == '__main__':
    train()
