import matplotlib.pyplot as plt

from .area_grabber import grab_area
from .element_grabber import grab_elements
from .classifier import classify
from .solver import solve, coverage_ratio


def check_image(path: str):
    print('Getting processable image area...', end='')
    image = grab_area(path)
    print('Complete!')

    print('Collecting elements...', end='')
    elements = grab_elements(image)
    print('Complete!')

    print('Classifying elements...', end='')
    objects, polygon = classify(elements)
    print('Complete!')
    print('Detected', len(objects), 'objects')

    print('Placing objects...')
    max_coverage = solve(objects, polygon)
    print('Complete!')

    print('R =', coverage_ratio(max_coverage, objects, polygon))
    plt.imshow(max_coverage)
    plt.show()
