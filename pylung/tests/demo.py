#prepare test.csv: 1nodule 1nomark 1nodule……

import sys
import os
import argparse
import numpy as np
import csv
import random

list1 = []

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='./output/', help='data path')
    return parser

#select bignodule from srcpath
def selectAnno(srcpath):
    anno_csv = file(srcpath+'anno.csv', 'rb')
    reader = csv.reader(anno_csv)
    csvfile = file('./tmp/nodule.csv', 'ab+')
    writer = csv.writer(csvfile)
    X, Y = [], []
    count=[0,0,0,0]  #nodule counter: 0:nomark, 1:big, 2:small, 3:nonodle
    for line in reader:
        if(int(line[1])==0):
            count[0]+=1
        elif(int(line[1])==1):
            count[1]+=1
            writer.writerow([srcpath,line[0],line[1]])
        elif(int(line[1])==2):
            count[2]+=1
            #writer.writerow([srcpath,line[0],line[1]])
        else:
            count[3]+=1
            #writer.writerow([srcpath,line[0],line[1]])
    return count

def selectPa(srcpath):
    pa_csv = file(srcpath+'pa.csv', 'rb')
    reader = csv.reader(pa_csv)
    csvfile = file('./tmp/nodule.csv', 'ab+')
    writer = csv.writer(csvfile)
    csvfile2 = file('./tmp/else.csv', 'ab+')
    writer2 = csv.writer(csvfile2)
    X, Y = [], []
    count=[0,0,0,0]
    for line in reader:
        if(int(line[1])==0):
            count[0]+=1
            writer2.writerow([srcpath,line[0],line[1]])
        elif(int(line[1])==1):
            count[1]+=1
            #writer.writerow([srcpath, line[0], line[1]])
        elif(int(line[1])==2):
            count[2]+=1
        else:
            count[3]+=1
    return count


def combine():
    csvfile = file('./tmp/nomark.csv', 'rb')
    reader = csv.reader(csvfile)
    csvfile2 = file('./tmp/nodule.csv', 'rb')
    reader2 = csv.reader(csvfile2)
    csvfile3 = file('test.csv', 'wb')
    writer = csv.writer(csvfile3)
    for a,b in zip(reader, reader2):
        writer.writerow([ a[0], a[1], a[2] ])
        writer.writerow([ b[0], b[1], b[2] ])

def select(length):
    csvfile = file('./tmp/else.csv', 'rb')
    reader = csv.reader(csvfile)
    csvfile2 = file('./tmp/nomark.csv', 'wb')
    writer = csv.writer(csvfile2)
    for line in reader:
        list1.append(line)
    print len(list1)
    random.shuffle(list1)
    list2 = list1[:length]
    for i in list2:
        writer.writerow([i[0],i[1],i[2]])
    return len(list2)



def main():
    parser = build_parser()
    options = parser.parse_args()
    length = 0
    #for file in os.listdir("./else/"):
        #file_path = "./else/"+file+"/"
        #count1 = selectAnno(file_path)
        #print "big-nodule:",count1[1]
        #length += count1[1]
        #count = selectPa(file_path)
        #length+=count[1]
        #print "no-mark:",count[0],"nodule:", count[1]

    #select same scale data:
    after = select(2105)
    print "after-select:", after
    #combine 1anno,1pa,1anno,1pa......
    combine()



if __name__ == '__main__':
    main()
