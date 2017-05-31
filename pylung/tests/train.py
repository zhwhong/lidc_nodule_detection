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
    csvfile = file('test.csv', 'rb') # control train set
    reader = csv.reader(csvfile)
    X, Y = [], []
    count = 0
    for line in reader:
        # imsave('./tmp/train_%s_%s' % (line[2], line[1]), gray_to_rgb(dcm_to_gray(imread(line[0] + line[1]))))
        X.append(gray_to_rgb(dcm_to_gray(imread(line[0]+line[1]))))
        Y.append([int(line[2])])
        count+=1
        # if count >= 5000:
        #     break
    Y = dense_to_one_hot(np.array(Y), 2)

    # train model
    model.fit(X, Y,validation_set=0.1, batch_size=128, n_epoch=100,shuffle=True)
    model.save('pylung2.model')

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
    cnt = 0
    for i in range(len(X)):
        if(Y2[i]==Z[i]):
            cnt+=1  #correct count

    print "correct:%d, wrong:%d, all:%d, accuracy:%f"%(cnt,len(X)-cnt,len(X),float(cnt)/len(X))



if __name__ == '__main__':
    train()
