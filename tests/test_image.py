"""
## Raster helping functions behaviour definition

"""

from pytest import approx
import numpy as np

from lib.image import draw_line


class Test__lines():
    """Basic test cases."""

    def test__partial_opacity__lighter(self):
        """
        It repeats the same line with partial opacity.
        The image should become sligtly ligther after each step.
        """
        img = np.zeros((11, 11))  # black background
        average = 0

        draw_line(img, (0, 0), (10, 10), value=1, opacity=.7)
        new_average = img.mean()
        assert img.ptp() == approx(1 - .3 ** 1)
        assert new_average > average

        draw_line(img, (0, 0), (10, 10), value=1, opacity=.7)
        new_average = img.mean()
        assert img.ptp() == approx(1 - .3 ** 2)
        assert new_average > average

        draw_line(img, (0, 0), (10, 10), value=1, opacity=.7)
        new_average = img.mean()
        assert img.ptp() == approx(1 - .3 ** 3)
        assert new_average > average

        draw_line(img, (0, 0), (10, 10), value=1, opacity=.7)
        new_average = img.mean()
        assert img.ptp() == approx(1 - .3 ** 4)
        assert new_average > average

    def test__partial_opacity__darker(self):
        """
        It repeats the same line with partial opacity.
        The image should become sligtly darker after each step.
        """
        img = np.ones((11, 11))  # white background
        average = 1

        draw_line(img, (0, 0), (10, 10), value=0, opacity=.7)
        new_average = img.mean()
        assert new_average < average

        draw_line(img, (0, 0), (10, 10), value=0, opacity=.7)
        new_average = img.mean()
        assert new_average < average

        draw_line(img, (0, 0), (10, 10), value=0, opacity=.7)
        new_average = img.mean()
        assert new_average < average

        draw_line(img, (0, 0), (10, 10), value=0, opacity=.7)
        new_average = img.mean()
        assert new_average < average

    def test__basic_lines_arimethic(self):
        """
        Verification of known wanted behaviour.
        """

        img = np.ones((5, 5))  # white background
        img[2, 2] = 0
        assert img.ptp() == 1

        img = np.ones((5, 5))  # white background
        draw_line(img, (0, 0), (4, 4), value=0, opacity=.5)
        assert img.ptp() == .5
        assert img[2, 2] == .5
        assert img[0, 1] == approx(.853, 1e-2)

        img = np.ones((5, 5))  # white background
        img[2, 2] = 0
        opacity = .5
        value = 1
        expected_value = img[2, 2] * opacity + value * (1 - opacity)
        draw_line(img, (0, 0), (4, 4), value=value, opacity=opacity)
        assert img[2, 2] == expected_value
        expected_value = 1 * opacity + value * (1 - opacity)
        assert img[0, 0] == expected_value
        assert img.ptp() == .5

        img = np.zeros((5, 5))  # black background
        img[2, 2] = 1
        opacity = .5
        value = 1
        expected_value = img[2, 2] * opacity + value * (1 - opacity)
        draw_line(img, (0, 0), (4, 4), value=value, opacity=opacity)
        assert img[2, 2] == expected_value
        assert img.ptp() == 1
        img[2, 2] = 0
        assert img.ptp() == .5
