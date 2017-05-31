import cv2
import numpy as np

from pylung.filters.filter import CandidateFilter
from pylung.utils import floodfill
from pylung.utils import morphology_open
from pylung.utils import switch_pixels


def _get_parenchyma_mask(image, threashold):
    mask = np.copy(image)
    # mask = np.copy(image).astype(np.uint8)  # this way may have some problem with parenchyma ???
    mask = morphology_open(mask)
    ret, mask = cv2.threshold(mask, threashold, 255, cv2.THRESH_BINARY_INV)

    # set margin to black
    h, w = mask.shape[:2]
    for i in range(h):
        for j in range(w):
            if ((i == 0 or j == 0 or i == h - 1 or j == w - 1) and
                        mask[i][j] != 0):
                mask = floodfill(mask, (i, j), 0)

    # fill holes in middle
    mask = floodfill(mask, (0, 0), -1)
    mask = switch_pixels(mask, 0, 255)
    mask = switch_pixels(mask, -1, 0)
    return mask


class ParenchymaFilter(CandidateFilter):
    # TODO(chwang) find reference for the HU value 625
    PARENCHYMA_THRESHOLD = 625

    def __init__(self, image):
        super(ParenchymaFilter, self).__init__(image)
        self._mask = _get_parenchyma_mask(image, self.PARENCHYMA_THRESHOLD)

    def _filter(self, x, y, h, w):
        centroid = [x + h / 2, y + w / 2]
        return self._mask[centroid[0]][centroid[1]] <= 0


filter_by_parenchyma = (lambda image, candidates:
                        ParenchymaFilter(image).filter(candidates))
