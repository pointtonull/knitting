import sys

import numpy as np

from lib.interface import showimage
from lib.image import draw_line


def main():
    print("Main function")
    img = np.ones((11, 11))
    draw_line(0, 0, 10, 10, img)
    draw_line(0, 10, 10, 0, img)
    draw_line(0, 10, 10, 0, img)
    draw_line(0, 10, 10, 0, img)
    draw_line(0, 5, 10, 5, img)
    draw_line(0, 5, 10, 5, img)
    draw_line(0, 5, 10, 5, img)
    draw_line(5, 0, 5, 10, img)
    print(img)
    showimage(img, "AA Lines with partial opacity")


if __name__ == "__main__":
    sys.exit(main())
