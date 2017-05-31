# Math (can still use '//' if want integer output; this library gives floats)
from __future__ import division

# General
import os, re
import math, cmath
import cStringIO, StringIO
from datetime import datetime
from flask import Flask, render_template, jsonify, redirect, url_for, request, send_file

# Image Processing
import numpy as np
from numpy.linalg import inv
import numpy.ma as ma
import cv2, cv
import matplotlib.pyplot as plt
import mahotas
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage.feature import blob_doh, peak_local_max

# Image Drawing
from PIL import ImageFont
from PIL import ImageDraw
from PIL import Image

# Storage
from boto.s3.connection import S3Connection
from boto.s3.key import Key

# SciPy
from scipy import ndimage
from scipy import signal
from scipy.signal import argrelextrema
from skimage import morphology
from skimage.morphology import watershed, disk, ball, cube, opening, octahedron
from skimage import data
from skimage.filter import rank, threshold_otsu, canny
from skimage.util import img_as_ubyte

#--------------------------#
#         GLOBALS          #
#--------------------------#

LOGFILE = 'pnod_results.txt'
FONT_PATH = '/Library/Fonts/'
DEBUG = False

app = Flask(__name__)
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads' # change to relative /var/www/ file

ALLOWED_EXTENSIONS = ['pdf', 'png', 'jpg', 'jpeg', 'gif', 'tif', 'tiff', 'dcm', 'dicom']
TIFF_EXTENSIONS = ['tif', 'tiff']
DICOM_EXTENSIONS = ['dcm', 'dicom']
PNG_EXTENSIONS = ['png']
JPG_EXTENSIONS = ['jpg', 'jpeg']
DEBUG = False
FONT_PATH = 'static/fonts/'
ACCESS_KEY = ''
SECRET_KEY = ''

@app.route('/')
def hello_world():
    print 'Hello World!'
    return render_template('index.html')

@app.route('/process_serve_chestct', methods=['GET'])
def process_serve_chestct():

    # Init
    imgfile = request.args.get('imgfile')
    print "Process/Serving Image: "+imgfile
    fnamesplit = imgfile.rsplit('.', 1)
    ext = fnamesplit[1]
    imgprefix = fnamesplit[0]

    prfxsplt = imgprefix.rsplit('-', 2)
    prefix = prfxsplt[0]
    nfiles = int(prfxsplt[1])
    idx = int(prfxsplt[2])

    matrix3D = np.array([])
    initFile = True
    imgarr = []

    conn = S3Connection(ACCESS_KEY, SECRET_KEY)
    bkt = conn.get_bucket('chestcad')
    k = Key(bkt)

    print prefix, str(nfiles)
    try:
        for x in range(0,nfiles):
            mykey = prefix+'-'+str(nfiles)+'-'+str(x)
            k.key = mykey

            # NEW: Process file here...
            fout = cStringIO.StringIO()
            k.get_contents_to_file(fout)
            if initFile:
                matrix3D = readChestCTFile(fout)
            else:
                matrix3D = np.concatenate((matrix3D,readChestCTFile(fout)), axis=2)
            initFile = False
        result = 1
    except:
        result = 0

    dimension = matrix3D.shape[0]
    panel1, panel2, panel3, panel4 = processMatrix(matrix3D, dimension)
    print 'prior to printing matrix'
    imgarr, data_encode = printMatrix(panel1, panel2, panel3, panel4, prefix, nfiles)

    return jsonify({"success":result,"imagefile":data_encode,"imgarr":imgarr})

@app.route('/upload', methods=['POST'])
def upload():

    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            now = datetime.now()

            # Naming and storage to S3 database
            prefix = file.filename.rsplit('.', 1)[0]

            conn = S3Connection(ACCESS_KEY, SECRET_KEY)
            bkt = conn.get_bucket('chestcad')
            k = Key(bkt)
            k.key = prefix
            if istiff(file.filename):
                k.set_contents_from_file(file, headers={"Content-Type":"image/tiff"})
            elif isjpg(file.filename):
                k.set_contents_from_file(file, headers={"Content-Type":"image/jpeg"})
            elif ispng(file.filename):
                k.set_contents_from_file(file, headers={"Content-Type":"image/png"})
            elif isdicom(file.filename):
                ds = dicom.read_file(file)
                pil_dcm = get_dicom_PIL(ds)
                pil_dcm_str = cStringIO.StringIO()
                pil_dcm.save(pil_dcm_str, format='tiff')
                pil_dcm_str.seek(0)
                k.set_contents_from_file(pil_dcm_str, headers={"Content-Type":"image/tiff"})
            else:
                k.set_contents_from_file(file) # don't suspect that this will work

            return jsonify({"success":True, "file": file.filename}) # passes to upload.js, function uploadFinished

#----------------------------#
#         FUNCTIONS          #
#----------------------------#

def get_label_size(a):
    return 1

def thresh(a, b, max_value, C):
    return max_value if a > b - C else 0

def block_size(size):
    block = np.ones((size, size), dtype='d')
    block[(size - 1 ) / 2, (size - 1 ) / 2] = 0
    return block


def custom_square(n):
    mycustom = np.ones((n,n,1), dtype=np.int)

    return mycustom

def custom_cross(n):
    mycustom = np.zeros((n,n,1), dtype=np.int)
    mycustom[0:n,1:n-1] = 1
    mycustom[1:n-1,0:n] = 1

    return mycustom

def get_number_neighbours(mask,block):
    '''returns number of unmasked neighbours of every element within block'''
    mask = mask / 255.0
    return signal.convolve2d(mask, block, mode='same', boundary='symm')
    #return signal.fftconvolve(mask, block, mode='same')

def masked_adaptive_threshold(image,mask,max_value,size,C):
    '''thresholds only using the unmasked elements'''
    block = block_size(size)
    conv = signal.convolve2d(image, block, mode='same', boundary='symm')

    #conv = signal.fftconvolve(image, block, mode='same')
    mean_conv = conv / get_number_neighbours(mask,block)

    return v_thresh(image, mean_conv, max_value,C)

def motsu(im,iz):
    th = mahotas.otsu(im, ignore_zeros = iz)
    int_out, binary_sum, intinv_out, binaryinv_sum = bintoint(im, th)

    return binary_sum, int_out, binaryinv_sum, intinv_out

def bintoint(im, thresh):
    bool_out = im > thresh
    binary_out = bool_out.astype('uint8')
    binaryinv_out = np.logical_not(bool_out).astype('uint8')
    #opened_out = ndimage.binary_opening(binary_out, structure=np.ones((2,2))).astype('uint8')
    tot = binary_out.sum()
    out = binary_out * 255

    totinv = binaryinv_out.sum()
    outinv = binaryinv_out * 255

    return out, tot, outinv, totinv

def lpf(im, s, o, d):
    # Apply Large Gaussian Filter To Cropped Image
    blur = ndimage.gaussian_filter(im, (s,s), order=o)/d

    # Apply Subtraction
    crpint = im.astype('int')
    subtracted = np.zeros_like(im)
    np.clip(crpint-blur, 0, 255, out=subtracted) # the choice between 0 and 1 is based on the OTSU calculation, or whether or not to include all fat pixels
    return subtracted, blur

# Same as LPF above but don't subtract values below a certain level
def lpf2(im, s, o, d):
    # Apply Large Gaussian Filter To Cropped Image
    blur = ndimage.gaussian_filter(im, (s,s), order=o)/d

    m = np.average(blur[blur[:,:] > 15])

    low_values_indices = blur < m  # Where values are low
    blur[low_values_indices] = 0

    # Apply Subtraction
    crpint = im.astype('int')
    subtracted = np.zeros_like(im)
    np.clip(crpint-blur, 0, 255, out=subtracted) # the choice between 0 and 1 is based on the OTSU calculation, or whether or not to include all fat pixels
    return subtracted, blur

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def isjpg(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in JPG_EXTENSIONS

def ispng(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in PNG_EXTENSIONS

def istiff(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in TIFF_EXTENSIONS

def isdicom(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in DICOM_EXTENSIONS

def readChestCTFile(f):

    # Read the file into an array
    array = np.frombuffer(f.getvalue(), dtype='uint8') # or use uint16?
    img = cv2.imdecode(array, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    imarray = np.array(img)

    imarray = denoise_tv_chambolle(imarray, weight=0.001, multichannel=False)
    imarray = (imarray*255).astype('uint8')

    slice3D = np.expand_dims(imarray, axis=2)

    return slice3D

def processMatrix(mtx,dim):

    #-------------------#
    #       LPF         #
    #-------------------#

    original = mtx

    # Apply Large Gaussian Filter To Cropped Image
    #gmtx = ndimage.gaussian_filter(mtx, (2,2,2), order=0)
    gmtx = mtx
    gmtx = (gmtx > gmtx.mean())
    gmtx = gmtx.astype('uint8')
    gmtx = gmtx * 255
    print gmtx.shape

    #------------------------#
    #        DENOISING       #
    #------------------------#
    erodim = 4 # erosion dimension
    cr = 3 # custom radius for the structured

    if dim == 512:
        erodim = 4
        cr = 3
    elif dim == 712:
        erodim = 5
        cr = 5

    gmtx_eroded = ndimage.binary_erosion(gmtx, structure=custom_cross(erodim)).astype(gmtx.dtype)
    #gmtx_eroded = ndimage.binary_erosion(gmtx, structure=myball).astype(gmtx.dtype)
    gmtx_eroded = gmtx_eroded.astype('uint8')
    eroded = gmtx_eroded * 255

    #gmtx_eroded = gmtx
    #eroded = gmtx_eroded

    #---------------------#
    #       SKIMAGE       #
    #---------------------#
    markers, nummarks = ndimage.label(gmtx_eroded)
    bins = np.bincount(markers.flatten())
    bins[0] = 0
    cwmark = np.argmax(bins)
    for zdim in xrange(markers.shape[2]):
        for col in xrange(markers.shape[1]):
            first = np.argmax(markers[:,col,zdim] == cwmark)
            last = markers.shape[1] - 1 - np.argmax(markers[::-1,col,zdim] == cwmark)
            markers[0:first,col,zdim] = cwmark
            markers[last:,col,zdim] = cwmark

    #markers = markers.astype('uint8')
    markers[markers == cwmark] = 0 # in markers image, 0 is background and 1 is the largest blob (i.e. chest wall & mediastinum)
    markers = markers > 0
    markers = markers.astype('uint8')

    myelem = custom_square(cr)
    opened = ndimage.morphology.binary_opening(markers, myelem)
    opened = opened.astype('uint8')

    markers, nummarks = ndimage.label(opened)
    opened = opened * 255

    bins = np.bincount(markers.flatten())

    for i in range(1, nummarks+1):
        if bins[i] > 10:
            com = ndimage.measurements.center_of_mass(markers == i)
            print com
            tmpimg_orig = np.array(original[:,:,int(com[2])])
            tmpimg_open = np.array(opened[:,:,int(com[2])])
            cv2.circle(tmpimg_orig,(int(com[1]),int(com[0])),50,[255,255,255],10)
            cv2.circle(tmpimg_open,(int(com[1]),int(com[0])),50,[255,255,255],10)
            original[:,:,com[2]] = tmpimg_orig
            opened[:,:,com[2]] = tmpimg_open

    return original, eroded, markers, opened

def printMatrix(p1, p2, p3, p4, pfx, nfls):

    BUFF = 40

    arr = []
    dataenc = None

    conn = S3Connection(ACCESS_KEY, SECRET_KEY)
    bkt = conn.get_bucket('chestcad')
    k = Key(bkt)

    numslices = p1.shape[2]

    if numslices != nfls:
        print 'The number of matrix slices (Z-dim) is not equal to the saved number of files.'

    for z in xrange(nfls):

        mykey = pfx+'-'+str(nfls)+'-'+str(z)
        k.key = mykey

        # Paste 4x4 Plot
        '''pil_panel1 = Image.fromarray(p1[:,:,z])
        pil_panel2 = Image.fromarray(p2[:,:,z])
        pil_panel3 = Image.fromarray(p3[:,:,z])
        pil_panel4 = Image.fromarray(p4[:,:,z])

        pil_panel1 = pil_panel1.convert('RGB')
        pil_panel2 = pil_panel2.convert('RGB')
        pil_panel3 = pil_panel3.convert('RGB')
        pil_panel4 = pil_panel4.convert('RGB')

        pw, ph = pil_panel1.size
        pil_backdrop = Image.new('RGB', (pw*2+BUFF*3,ph*2+BUFF*3), "white")

        pil_backdrop.paste(pil_panel1,(BUFF,BUFF))
        pil_backdrop.paste(pil_panel2,(pw+BUFF*2,BUFF))
        pil_backdrop.paste(pil_panel3,(BUFF,BUFF*2+ph))
        pil_backdrop.paste(pil_panel4,(pw+BUFF*2,BUFF*2+ph))

        pil_backdrop.save(outdir+pre+str(z)+'.png')'''

        copyarr = p1[:,:,z].copy()
        pil_panel1 = Image.fromarray(copyarr)
        pil_panel1 = pil_panel1.convert('RGB')

        imgout = cStringIO.StringIO()
        pil_panel1.save(imgout, format='png')
        k.set_contents_from_string(imgout.getvalue())
        imgout.close()

        data = k.get_contents_as_string()
        #k.delete() # putting the delete here causes premature loss of the image; need to find somewhere else to do it probably performed via outside function when called from javascript
        dataenc = data.encode("base64")
        arr.append(k.generate_url(3600))

    return arr, dataenc

def get_LUT_value(data, window, level):
    return np.piecewise(data,
                        [data <= (level - 0.5 - (window - 1) / 2),
                         data > (level - 0.5 + (window - 1) / 2)],
                        [0, 255, lambda data: ((data - (level - 0.5)) / (window - 1) + 0.5) * (255 - 0)])

# Display an image using the Python Imaging Library (PIL)
def get_dicom_PIL(dataset):
    if ('PixelData' not in dataset):
        raise TypeError("Cannot show image -- DICOM dataset does not have pixel data")
    if ('WindowWidth' not in dataset) or ('WindowCenter' not in dataset):  # can only apply LUT if these values exist
        bits = dataset.BitsAllocated
        samples = dataset.SamplesPerPixel
        if bits == 8 and samples == 1:
            mode = "L"
        elif bits == 8 and samples == 3:
            mode = "RGB"
        elif bits == 16:
            mode = "I;16"  # not sure about this -- PIL source says is 'experimental' and no documentation. Also, should bytes swap depending on endian of file and system??
        else:
            raise TypeError("Don't know PIL mode for %d BitsAllocated and %d SamplesPerPixel" % (bits, samples))

        # PIL size = (width, height)
        size = (dataset.Columns, dataset.Rows)

        im = Image.frombuffer(mode, size, dataset.PixelData, "raw", mode, 0, 1)  # Recommended from the original code to specify all details by http://www.pythonware.com/library/pil/handbook/image.html; this is experimental...
    else:
        image = get_LUT_value(dataset.pixel_array, dataset.WindowWidth, dataset.WindowCenter)
        im = Image.fromarray(np.uint8(image)).convert('L')  # Convert mode to L since LUT has only 256 values: http://www.pythonware.com/library/pil/handbook/image.htm

    return im
