from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import binary_fill_holes
from skimage.color import rgb2gray
from skimage.morphology import binary_closing
from skimage.feature import canny
from imageio import imread


def __crop_horizontal(edge_segmentation: np.ndarray) -> Tuple[float, float]:
    top_border, low_border = 0, len(edge_segmentation)
    for i, row in enumerate(edge_segmentation):
        if row[0] is np.True_ and row[-1] is np.True_:
            if np.False_ in row:
                continue
            top_border = i
            break

    for i, row in enumerate(reversed(edge_segmentation)):
        if row[0] is np.True_ and row[-1] is np.True_:
            if np.False_ in row:
                continue
            low_border = len(edge_segmentation) - i
            break
    return top_border, low_border


def __crop_vertical(edge_segmentation: np.ndarray) -> Tuple[float, float]:
    edge_segmentation_transposed = np.transpose(edge_segmentation)

    left_border, right_border = 0, len(edge_segmentation_transposed)
    for i, row in enumerate(edge_segmentation_transposed):
        if row[0] is np.True_ and row[-1] is np.True_:
            if np.False_ in row:
                continue
            left_border = i
            break

    for i, row in enumerate(reversed(edge_segmentation_transposed)):
        if row[0] is np.True_ and row[-1] is np.True_:
            if np.False_ in row:
                continue
            right_border = len(edge_segmentation_transposed) - i
            break
    return left_border, right_border


def grab_area(path: str, verbose: bool = False) -> np.ndarray:
    raw_image = imread(path)

    raw_image_gray = rgb2gray(raw_image)
    canny_image = canny(raw_image_gray)

    edge_map = binary_closing(canny_image, footprint=np.ones((11, 11)))
    edge_segmentation = binary_fill_holes(edge_map)

    top_border, low_border = __crop_horizontal(edge_segmentation)
    left_border, right_border = __crop_vertical(edge_segmentation)

    crop_image = raw_image[top_border:low_border, left_border:right_border]

    if verbose:
        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(raw_image)
        ax[0].set_title("Original image")
        ax[0].set_axis_off()

        ax[1].imshow(crop_image)
        ax[1].set_title("Area od interest")
        ax[1].set_axis_off()

        plt.show()

    return crop_image
