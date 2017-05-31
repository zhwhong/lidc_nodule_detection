import numpy as np
from struct import *
import scipy.misc


def main():
    h, w, l = 133, 512, 512
    data = [[[0 for i in range(l)] for j in range(w)] for k in range(h)]
    file = open(r"instance1.dat", "rb")
    for i in range(h):
        print 'processing image %d / %d...' % (i,h-1)
        for j in range(w):
            for k in range(l):
                data[i][j][k], = unpack("h", file.read(2))
    data = np.array(data)
    print data.shape
    for i in range(h):
        print 'save image %d / %d...' % (i,h-1)
        scipy.misc.imsave('./output/out_%d.png' % (i,), data[i])
    file.close()

"""
import dicom
import numpy as np
import struct
import ctypes
import scipy.misc

def main():
    dcm = dicom.read_file('./test/1.dcm')
    image = np.array(dcm.pixel_array)
    print image.shape
    h, w = image.shape[:2]
    img = image.reshape(h*w)
    print img.shape
    # file = open(r"out.dat", "wb")
    s = struct.Struct('262144h')
    prebuffer = ctypes.create_string_buffer(s.size)
    s.pack_into(prebuffer, 0, *img)
    result = s.unpack_from(prebuffer, 0)
    result = np.array(result)
    print result.shape
    result = result.reshape(512,512)
    print result.shape
    scipy.misc.imsave('out.png',result)
    # file.close()
"""

if __name__ == '__main__':
    main()