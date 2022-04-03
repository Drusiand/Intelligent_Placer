import numpy as np
import matplotlib.pyplot as plt
import cv2

from scipy.ndimage import binary_closing, binary_fill_holes
from skimage.color import label2rgb
from typing import List, Tuple

from skimage.morphology import binary_dilation, binary_erosion


def get_mask_from_canny(canny_image: np.ndarray) -> np.ndarray:
    # generating mask
    edge_map = binary_closing(canny_image)
    edge_segmentation = binary_fill_holes(edge_map)

    # filtering noises
    edge_segmentation = binary_erosion(edge_segmentation, footprint=np.ones((5, 5)))
    edge_segmentation = binary_dilation(edge_segmentation, footprint=np.ones((5, 5)))

    return edge_segmentation


def get_dominant_color(image: np.ndarray) -> np.ndarray:
    # count mean values on each channel
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # average = image_gray.mean(axis=0).mean(axis=0)
    pixels = np.float32(image_gray.reshape(-1))

    # as image is grayscale
    n_colors = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]

    return dominant


def classify(images: List[np.ndarray], verbose: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    polygon_index = 0
    masks = []
    objects, polygons = [], []
    max_white = 0

    for i, image in enumerate(images):
        processable_image = cv2.addWeighted(image, 1.25, image, 0, -100)
        canny_image = cv2.Canny(processable_image, 150, 300)  # Empirical parameters

        masks.append(get_mask_from_canny(canny_image))

        dominant = get_dominant_color(processable_image)

        if sum(dominant) > max_white:
            max_white = dominant
            polygon_index = i

        if verbose:
            fig, ax = plt.subplots(1, 4, figsize=(15, 6))

            ax[0].imshow(image)
            ax[0].set_title("Original image")
            ax[0].set_axis_off()

            ax[1].imshow(processable_image)
            ax[1].set_title("Processed image")
            ax[1].set_axis_off()

            ax[2].imshow(canny_image, cmap='gray')
            ax[2].set_title("Canny edges")
            ax[2].set_axis_off()

            ax[3].set_title("Edge-based segmentation")
            ax[3].imshow(label2rgb(masks[-1], image=image))
            ax[3].set_axis_off()

            plt.tight_layout()
            plt.show()

    for j, mask in enumerate(masks):
        if j == polygon_index:
            polygons.append(mask)
        else:
            objects.append(mask)

    if verbose:

        fig, axes = plt.subplots(1, len(images))

        for i, obj in enumerate(objects):
            axes[i].set_title('object ' + str(i))
            axes[i].set_axis_off()
            axes[i].imshow(obj)

        axes[-1].set_title('polygon')
        axes[-1].set_axis_off()
        axes[-1].imshow(polygons[0])

        plt.tight_layout()
        plt.show()

    return objects, polygons
