import cStringIO
import os

import cv2
import dicom
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage.restoration import denoise_tv_chambolle


def read_ct_file(dicom_file):
    # Read the file into an array
    dcm = dicom.read_file(dicom_file)
    img = dcm.pixel_array
    imarray = np.array(img)

    imarray = denoise_tv_chambolle(imarray, weight=0.001, multichannel=False)
    imarray = (imarray * 255).astype('uint8')

    slice3D = np.expand_dims(imarray, axis=2)

    return slice3D


def custom_square(n):
    mycustom = np.ones((n, n, 1), dtype=np.int)

    return mycustom


def custom_cross(n):
    mycustom = np.zeros((n, n, 1), dtype=np.int)
    mycustom[0:n, 1:n - 1] = 1
    mycustom[1:n - 1, 0:n] = 1

    return mycustom


def process_matrix(mtx):
    # -------------------#
    #       LPF         #
    # -------------------#

    original = mtx
    dim = mtx.shape[0]

    # Apply Large Gaussian Filter To Cropped Image
    # gmtx = ndimage.gaussian_filter(mtx, (2,2,2), order=0)
    gmtx = np.copy((mtx > mtx.mean()) * 255).astype(np.uint16)
    print('gmtx:', gmtx.shape)

    # ------------------------#
    #        DENOISING       #
    # ------------------------#
    erodim = 4  # erosion dimension
    cr = 3  # custom radius for the structured

    if dim == 512:
        erodim = 4
        cr = 3
    elif dim == 712:
        erodim = 5
        cr = 5

    gmtx_eroded = ndimage.binary_erosion(gmtx,
                                         structure=custom_cross(erodim)).astype(
        gmtx.dtype)
    # gmtx_eroded = ndimage.binary_erosion(gmtx, structure=myball).astype(gmtx.dtype)
    gmtx_eroded = gmtx_eroded.astype('uint8')
    eroded = gmtx_eroded * 255

    # gmtx_eroded = gmtx
    # eroded = gmtx_eroded

    # ---------------------#
    #       SKIMAGE       #
    # ---------------------#
    markers, nummarks = ndimage.label(gmtx_eroded)
    bins = np.bincount(markers.flatten())
    bins[0] = 0
    cwmark = np.argmax(bins)
    for zdim in xrange(markers.shape[2]):
        for col in xrange(markers.shape[1]):
            first = np.argmax(markers[:, col, zdim] == cwmark)
            last = markers.shape[1] - 1 - np.argmax(
                markers[::-1, col, zdim] == cwmark)
            markers[0:first, col, zdim] = cwmark
            markers[last:, col, zdim] = cwmark

    # markers = markers.astype('uint8')
    markers[
        markers == cwmark] = 0  # in markers image, 0 is background and 1 is the largest blob (i.e. chest wall & mediastinum)
    markers = markers > 0
    markers = markers.astype('uint8')

    myelem = custom_square(cr)
    opened = ndimage.morphology.binary_opening(markers, myelem)
    opened = opened.astype('uint8')

    markers, nummarks = ndimage.label(opened)
    opened = opened * 255

    bins = np.bincount(markers.flatten())

    for i in range(1, nummarks + 1):
        if bins[i] > 10:
            com = ndimage.measurements.center_of_mass(markers == i)
            print('com:', com)
            tmpimg_orig = np.array(original[:, :, int(com[2])])
            tmpimg_open = np.array(opened[:, :, int(com[2])])
            cv2.circle(tmpimg_orig, (int(com[1]), int(com[0])), 50,
                       [255, 255, 255], 10)
            cv2.circle(tmpimg_open, (int(com[1]), int(com[0])), 50,
                       [255, 255, 255], 10)
            original[:, :, com[2]] = tmpimg_orig
            opened[:, :, com[2]] = tmpimg_open

    return [original, eroded, markers, opened]


def printMatrix(p1, p2, p3, p4, nfls):
    BUFF = 40

    arr = []
    dataenc = None

    numslices = p1.shape[2]

    if numslices != nfls:
        print 'The number of matrix slices (Z-dim) is not equal to the saved number of files.'

    for z in xrange(nfls):
        mykey = str(nfls) + '-' + str(z)

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

        copyarr = p1[:, :, z].copy()
        pil_panel1 = Image.fromarray(copyarr)
        pil_panel1 = pil_panel1.convert('RGB')

        imgout = cStringIO.StringIO()
        pil_panel1.save(imgout, format='png')
        imgout.close()

    return arr, dataenc


file = os.path.join(os.path.dirname(__file__), '../tests/ct.dcm')


def get_LUT_value(data, window, level):
    return np.piecewise(data,
                        [data <= (level - 0.5 - (window - 1) / 2),
                         data > (level - 0.5 + (window - 1) / 2)],
                        [0, 255, lambda data: ((data - (level - 0.5)) / (
                            window - 1) + 0.5) * (255 - 0)])


# Display an image using the Python Imaging Library (PIL)
def get_dicom_PIL(dataset):
    if ('PixelData' not in dataset):
        raise TypeError(
            "Cannot show image -- DICOM dataset does not have pixel data")
    if ('WindowWidth' not in dataset) or (
                'WindowCenter' not in dataset):  # can only apply LUT if these values exist
        print('not lut')
        bits = dataset.BitsAllocated
        samples = dataset.SamplesPerPixel
        if bits == 8 and samples == 1:
            mode = "L"
        elif bits == 8 and samples == 3:
            mode = "RGB"
        elif bits == 16:
            mode = "I;16"  # not sure about this -- PIL source says is 'experimental' and no documentation. Also, should bytes swap depending on endian of file and system??
        else:
            raise TypeError(
                "Don't know PIL mode for %d BitsAllocated and %d SamplesPerPixel" % (
                    bits, samples))

        # PIL size = (width, height)
        size = (dataset.Columns, dataset.Rows)

        im = Image.frombuffer(mode, size, dataset.PixelData, "raw", mode, 0,
                              1)  # Recommended from the original code to specify all details by http://www.pythonware.com/library/pil/handbook/image.html; this is experimental...
    else:
        print('lut')
        image = get_LUT_value(dataset.pixel_array, dataset.WindowWidth,
                              dataset.WindowCenter)
        im = Image.fromarray(np.uint8(image)).convert(
            'L')  # Convert mode to L since LUT has only 256 values: http://www.pythonware.com/library/pil/handbook/image.htm

    return im


dcm = dicom.read_file(file)
im = get_dicom_PIL(dcm)
plt.imshow(im, 'gray')
plt.show()
exit(0)

nfiles = 1
matrix3D = read_ct_file(file)
# print(matrix3D)
# print(matrix3D.shape)
dimension = matrix3D.shape[0]
panels = process_matrix(matrix3D)


# imgarr, data_encode = printMatrix(panel1, panel2, panel3, panel4, nfiles)
#
# print('imgarr', imgarr)
# print('data_encode', data_encode)


def plot_panels(panels):
    n = len(panels)
    cols = 4
    plt.subplot((n + cols - 1) / cols, cols, 1)
    for i in range(n):
        panel = np.reshape(panels[i], (512, 512))
        plt.subplot((n + cols - 1) / cols, cols, i + 1)
        plt.imshow(panel, 'gray')
        # plt.title(self.MASKS[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


plot_panels(panels)
