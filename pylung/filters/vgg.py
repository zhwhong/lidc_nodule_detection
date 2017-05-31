import numpy as np

from pylung.filters.filter import CandidateFilter
from pylung.utils import dcm_to_gray, gray_to_rgb

import sys
# sys.path.append('../../../../easytf/')
import easytf


class VGGFilter(CandidateFilter):
    def __init__(self, image):
        super(VGGFilter, self).__init__(image)
        self._image = image
        # self._classifier = easytf.load_model('pylung.models.vgg_classifier', checkpoint='pylung2.model')
        self._classifier = easytf.load_model_by_module_name('pylung.models.vgg_classifier', checkpoint='pylung2.model')

    def _filter(self, x, y, h, w):
        # print(x, y, h, w)
        piece = self._image[x:x + h, y:y + w]
        # print(piece)
        piece = gray_to_rgb(dcm_to_gray(piece))
        cls = np.argmax(self._classifier.predict([piece]))
        print((x, y, h, w), cls)
        return cls == 0


filter_by_vgg = lambda image, candidates: VGGFilter(image).filter(candidates)
