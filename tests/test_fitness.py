import unittest
from pytest import approx

"""

## Fitness criteria

    - Knitter function must address `moire` issues.
    - Unfocus filter must be applied before comparation based on know dimensions of
      the frame and distance of the observer
    - Linear transformations should have limited effect on fitness
        - adjust brightness and contrast before comparation
        - adjusted values for brightness and contrast are used as part of fitness
          output

## Knitter function

    - returns a image of similar dimensions than the given one
    - avoids `moire patterns` formation

"""

from lib import fitness


class Naive():
    """Basic test cases."""

    def test__interface(self):
        assert isinstance(fitness.naive, float)
