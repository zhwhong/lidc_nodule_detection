import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


def floodfill(image, start_point, value):
    height, width = image.shape[:2]
    points = [start_point]
    flag = [[0 for j in range(width)] for i in range(height)]
    flag[start_point[0]][start_point[1]] = 1
    origin_value = image[start_point[0]][start_point[1]]
    while len(points) > 0:
        pt = points.pop(0)
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        for x, y in zip(dx, dy):
            if (0 <= pt[0] + x < height and 0 <= pt[1] + y < width and
                        origin_value == image[pt[0] + x][pt[1] + y] and
                        flag[pt[0] + x][pt[1] + y] == 0):
                flag[pt[0] + x][pt[1] + y] = 1
                points.append((pt[0] + x, pt[1] + y))
        image[pt[0]][pt[1]] = value
    return image


def switch_pixels(image, origin_value, value):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] == origin_value:
                image[i][j] = value
    return image


def morphology_open(image):
    # morphology open operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image


def morphology_close(image):
    # morphology close operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image


def gray_to_rgb(image):
    assert len(image.shape) == 2, 'Image is not grayscale'
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return rgb


def imsave(fname, image):
    figure = plt.figure()
    plt.imshow(image, 'gray')
    plt.xticks([]), plt.yticks([])
    figure.savefig(fname)


def dcm_to_gray(image):
    maxval = np.max(image)
    minval = np.min(image)
    valrange = float(maxval) - float(minval)
    return ((image.astype(np.float16) - minval) / valrange * 255).astype(
        np.uint8)


def find_all_files(root, suffix=None):
    res = []
    for root, _, files in os.walk(root):
        for f in files:
            if suffix is not None and not f.endswith(suffix):
                continue
            res.append(os.path.join(root, f))
    return res


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
