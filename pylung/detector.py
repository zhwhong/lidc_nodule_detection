import logging

import dicom
import numpy as np
from pylung.filters.parenchyma import filter_by_parenchyma
from pylung.filters.vgg import filter_by_vgg
from pylung.utils import imsave
import scipy.misc


class NoduleDetector(object):
    def __init__(self, dicom_image):
        if dicom_image is None:
            raise TypeError('Dicom image is None.')

        dcm = dicom.read_file(dicom_image)
        logging.debug('Dicom modality: %s' % dcm.get('Modality'))
        logging.debug('Dicom body part: %s' % dcm.get('BodyPartExamined'))
        if ((dcm.get('BodyPartExamined').upper() not in ('CHEST', 'LUNG')
             or dcm.get('Modality').upper() != 'CT')):
            raise TypeError('Only support lung CT image.')
        self._image = np.array(dcm.pixel_array)
        self._candidates = self._init_candidates()

    @property
    def candidates(self):
        return self._candidates

    def _init_candidates(self):
        h, w = self._image.shape[:2]
        pieces = []
        window = [64, 64]
        for x in range(0, h - window[0] / 2, window[0] / 2):
            for y in range(0, w - window[1] / 2, window[1] / 2):
                pieces.append([x, y, window[0], window[1]])
        return pieces

    def detect(self):
        self._candidates = filter_by_parenchyma(self._image, self._candidates)
        self._candidates = filter_by_vgg(self._image, self._candidates)
        for x, y, h, w in self._candidates:
            # imsave('o/v_%d_%d.jpg' % (x, y), self._image[x:x + h, y:y + w])
            scipy.misc.imsave('o/v_%d_%d.jpg' % (x, y), self._image[x:x + h, y:y + w])
        return self._candidates

    def get_parenchyma_block(self):
        return filter_by_parenchyma(self._image, self._candidates)

detect_nodule = (lambda dicom_image: NoduleDetector(dicom_image).detect())
parenchyma_block = (lambda dicom_image: NoduleDetector(dicom_image).get_parenchyma_block())
