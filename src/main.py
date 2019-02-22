import sys

import numpy as np

from lib.interface import showimage
from lib.image import draw_line


def main():
    print("Main function")
#     img = np.ones((11, 11))
    img = np.zeros((11, 11))
    img[5, 5] = 1

    draw_line(img,  (0,  0),  (10,  10), value=.5, opacity=.01)
    draw_line(img,  (0,  10), (10,  0),  value=.5, opacity=.01)
    draw_line(img,  (0,  10), (10,  0),  value=.5, opacity=.01)
    draw_line(img,  (0,  10), (10,  0),  value=.5, opacity=.01)
    draw_line(img,  (0,  5),  (10,  5),  value=.5, opacity=.01)
    draw_line(img,  (0,  5),  (10,  5),  value=.5, opacity=.01)
    draw_line(img,  (0,  5),  (10,  5),  value=.5, opacity=.01)
    draw_line(img,  (5,  0),  (5,   10), value=.5, opacity=.01)

    print(img)
    print(img.ptp())
    showimage(img, "AA Lines with partial opacity")


if __name__ == "__main__":
    sys.exit(main())
