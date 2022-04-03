from typing import List

import matplotlib.pyplot as plt
import numpy as np
import cv2

OFFSET = 5  # offset for detected cropped objects
H_BORDER = 50
W_BORDER = 50


# use image_index only for saving elements!
def grab_elements(image: np.ndarray, verbose: bool = False, image_index: bool = None) -> List[np.ndarray]:
    detected_elements = []

    # processable_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    processable_image = image.copy()
    canny_image = cv2.Canny(processable_image, 100, 500)  # empirical parameters!!!

    # applying closing function
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(canny_image, cv2.MORPH_CLOSE, kernel)

    # finding_contours
    (contours, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_index = 0
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width > W_BORDER and height > H_BORDER:
            contour_index += 1
            new_img = image[y - OFFSET:y + height + OFFSET, x - OFFSET:x + width + OFFSET]
            if verbose:
                if image_index is not None:
                    print('saved', str(image_index) + '_' + str(contour_index) + '.png')
                    cv2.imwrite(str(image_index) + '_' + str(contour_index) + '.png', new_img)
            detected_elements.append(new_img)

    if verbose:
        fig, axes = plt.subplots(1, len(detected_elements))

        fig.suptitle('detected elements')
        for ax, image in zip(axes, detected_elements):
            ax.imshow(image)
            ax.set_axis_off()
        plt.tight_layout()
        plt.show()

    return detected_elements
