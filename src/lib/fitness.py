"""
A good portrait is one the viewer easily identify with the subject.
That's why I decided to base the fitness functions on human eye/brain features.

- Knitter function must address `moire` issues.
- Unfocus filter must be applied before comparation based on know dimensions of
  the frame and distance of the observer
- Linear transformations should have limited effect on fitness
    - adjust brightness and contrast before comparation
    - adjusted values for brightness and contrast are used as part of fitness
      output

## Knitter function

This is the procedure that creates the image, given target image and weights.

Features:

    - returns a image of similar dimensions than the given one
    - avoids `moire patterns` formation

# TODO:

    - Create fitness criteria tests
    - TDD fitness functions
    - Create variable-deep thread finder
    - Create Newton fast approximation for smart cropping
    - Use Newton weights to prioritize thread finder
"""

from . import image


def naive(left, right):
    """
    Sum of difference between target and pilot.
    """
    img_diff = image.subtract(left, right) ** 2
    return img_diff.sum()


def visual(target, render):
    pass
    return 0


def default(target, render):
    return naive(target, render)

