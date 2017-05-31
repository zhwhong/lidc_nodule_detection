import sys
import json
import string
import random

if len(sys.argv) != 4:
    print ":  python convert_txt2json.py lung_train.txt lung_train.json 2"
    exit()

json_images = []

path = sys.argv[1]
outfile_name = sys.argv[2]
rate = float(sys.argv[3])

pos_num = 0
neg_num = 0

with open(path, 'r') as f:
    done = 0
    while not done:
        aLine = f.readline()
        if (aLine != ''):
            filename = aLine
            filename = filename.strip('\n')
            rects = []
            num = f.readline()
            num = string.atoi(num)

            for i in range(0, num, 1):
                aLine = f.readline()

            if(num > 0) :
                pos_num = pos_num + 1
            else:
                neg_num = neg_num + 1
        else:
            done = 1


outfile = open(outfile_name, 'w')
with open(path, 'r') as f:
    done = 0
    while not done:
        aLine = f.readline()
        if (aLine != ''):
            filename = aLine
            filename = filename.strip('\n')
            rects = []
            num = f.readline()
            num = string.atoi(num)
            for i in range(0, num, 1):
                aLine = f.readline()
                splite_str = aLine.split()
                splite_int = [int(s) for s in splite_str]
                kind = splite_int[0]

                pad_x = 0
                if(splite_int[3]<32):
                    pad_x = (32 - splite_int[3])/2

                pad_y = 0
                if(splite_int[4]<32):
                    pad_y = (32 - splite_int[4])/2

                l = max(0, splite_int[1] - pad_x)
                r = min(351, splite_int[1] + splite_int[3] + pad_x)
                
                t = max(0, splite_int[2] - pad_y)
                b = min(255, splite_int[2] + splite_int[4] + pad_y)


                #bbox = dict([("x1", splite_int[1]), ("y1", splite_int[2]), ("x2", splite_int[1] + splite_int[3]), ("y2", splite_int[2] + splite_int[4])])
                bbox = dict([("x1", l), ("y1", t), ("x2", r), ("y2", b)])
                if(kind == 1):
                    rects.append(bbox)

            if(num > 0 & len(rects) > 0):
                json_image = dict([("image_path", filename), ("rects", rects)])
                json_images.append(json_image)
            elif(random.uniform(0, 1) <= rate*pos_num/neg_num):
                json_image = dict([("image_path", filename), ("rects", rects)])
                json_images.append(json_image)
        else:
            done = 1


outfile.write(json.dumps(json_images, indent = 1))
# outfile.write(json.dumps(json_images, indent=2, sort_keys=True))
outfile.close()
