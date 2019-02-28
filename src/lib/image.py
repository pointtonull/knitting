#!/usr/bin/env python

from functools import lru_cache
import itertools
import operator
import random

from PIL import Image as pil
from PIL import ImageOps
import numpy as np

from numpy import sin, cos, exp, log, pi
from scipy import misc, ndimage
from scipy.misc import imresize, imsave
from scipy.ndimage import geometric_transform, gaussian_filter
from skimage import draw
from skimage.feature import canny

from . import cache
from .interface import imshow, progressbar
from .minimize import generic_minimizer

VERBOSE = 0
tau = pi * 2  # twice as sexy as pi


def blur(image, sigma):
    """
    Just a wrapper around the choosen blur filter.
    """
    return gaussian_filter(image, sigma)


class Frame:

    def __init__(self, side, pins=200):
        """
        It assumes center is in (radius, radius) (i.e.: the borders are
        touching row 0 and column 0.
        """
        if type(side) is np.ndarray:
            side = min(side.shape)
        self.side = side
        self.center = side // 2 - 1 + side % 2
        self.radius = (side - 1) // 2
        self.pins = pins

    def get_pin_pos(self, pin_number, antimoire=3):
        """
        Since this coordinates are to be used as indexes they are always
        rounded to ceil integer.
        """
        angle = (tau / self.pins) * (- pin_number) - pi
        row = int(cos(angle) * self.radius) + self.center
        col = int(sin(angle) * self.radius) + self.center
        if antimoire:
            row += random.randrange(antimoire) - (antimoire // 2)
            row = max(0, row)
            row = min(self.side - 1, row)
            col += random.randrange(antimoire) - (antimoire // 2)
            col = max(0, col)
            col = min(self.side - 1, col)
        return row, col

    def get_segment_pixels(self, pin_from, pin_to, antiaaliasing=False):
        """
        Returns (rows, cols) covered by the segment.
        """
        if antiaaliasing:
            raise NotImplementedError("Not sure if needed.")
        r0, c0 = self[pin_from]
        r1, c1 = self[pin_to]
        return draw.line(r0, c0, r1, c1)

    def get_mask(self):
        """
        Returns a mask selecting the interior covered by the circle.
        """
        shape = (self.side, self.side)
        rows, cols = draw.circle(self.center, self.center, self.radius, shape)
        return rows, cols

    def segments(self):
        """
        Returts iterator over segments initial - final pin.
        """
        return itertools.combinations(range(self.pins), 2)

    @lru_cache(maxsize=None)
    def render_segment(self, segment, value=0, opacity=.5, blur_sigma=1):
        img_render = np.ones((self.side, self.side))
        pin_from, pin_to = segment
        draw_line(img_render, self[pin_from], self[pin_to],
                  opacity=opacity, fast=False)
        if blur_sigma:
            img_render = blur(img_render, blur_sigma)
        return img_render

    def render_weights(self, weights, draw_pins=False, fast=False, blur_sigma=1):
        """
        Creates a fast representation of the knitting.
        """
        if draw_pins:
            raise NotImplementedError("Generic exception")

        img_render = np.ones((self.side, self.side))
        segments_weights = zip(self.segments(), weights)
        for (pin_from, pin_to), opacity in segments_weights:
            draw_line(img_render, self[pin_from], self[pin_to],
                      opacity=opacity, fast=fast)

        if blur_sigma:
            img_render = blur(img_render, blur_sigma)

        return img_render

    def render_steps(self, steps, draw_pins=False, fast=False, blur_sigma=1,
                     opacity=.5, img_render=None):
        """
        Creates a fast representation of the knitting.
        """
        if draw_pins:
            raise NotImplementedError("Generic exception")

        if img_render is None:
            img_render = np.ones((self.side, self.side))

        if len(steps) < 2:
            raise ValueError("It is required at least to pins to create a "
                             "segment.")

        segments = zip(steps[:-1], steps[1:])
        for pin_from, pin_to in segments:
            draw_line(img_render, self[pin_from], self[pin_to],
                      opacity=opacity, fast=fast)

        if blur_sigma:
            img_render = blur(img_render, blur_sigma)

        return img_render

    def __mul__(self, value):
        return self.get_mask() * value

    def __getitem__(self, key):
        return self.get_pin_pos(key)


def draw_line(canvas, start, end, value=0, opacity=.6, fast=False):
    """
    In place draws a line with anti-aliasing and given opacity.

    canvas: array where to apply the line in-place.
    start: Initial row and column.
    end: Final row and column.
    value: color to paint.
    opacity: how much to affect the image with the new values.
    fast: it'll would execute a single operation, this ignores anti-alianing.
    """
    if fast:
        rr, cc = draw.line(*start, *end)
        original = canvas[rr, cc]
        canvas[rr, cc] = value * opacity + original * (1 - opacity)
    else:
        rr, cc, alpha = draw.line_aa(*start, *end)
        original = canvas[rr, cc]
        painted = value * alpha + original * (1 - alpha)
        canvas[rr, cc] = painted * opacity + original * (1 - opacity)


def auto_canny(array, average=None, gaussian_sigma=1, strongness=2.5,
               epsilon=0.0001):
    if average is None:
        average = array.size ** 0.5 / array.size
    array -= array.min()
    array /= array.max()

    def canny_average(hard_threshold):
        soft_threshold = hard_threshold / strongness
        edges = canny(array, gaussian_sigma, hard_threshold, soft_threshold)
        return edges.mean()

    hard_threshold = 0.4
    bottom, top = 0., 1.
    for iteration in range(20):
        current_average = canny_average(hard_threshold)
        print(hard_threshold, current_average)
        if abs(current_average - average) < epsilon:
            break
        elif current_average < average:
            top = hard_threshold
            hard_threshold = (bottom + top) / 2
        else:
            bottom = hard_threshold
            hard_threshold = (bottom + top) / 2
    else:
        print("Agotados los intentos")

    soft_threshold = hard_threshold / strongness
    return canny(array, gaussian_sigma, hard_threshold, soft_threshold)


@cache.hybrid
def get_subtract_paramns(left, right):
    """
    Returns k that minimizes:

        var(left - k * right)
    """

    def diference(k):
        return (left - k * right).var()

    best_k = float(generic_minimizer(diference, 1))
    return best_k


def subtract(left, right):
    """
    Will operate
        left - k * right + l

    Where k and l are the values that minimizes the result.
    """

    if right is None:
        return left
    else:
        best_k = get_subtract_paramns(left, right)
        result = left - best_k * right

        return result


def limit_size(image, limit, avoidodds=True):
    """
    Image is a numpy array.
    Resulution is a quantity:
        in pixels if >= 1000
        in megapixels if < 1000
    """

    if limit < 1000:
        limit *= 1e6

    relsize = (image.size / limit) ** -.5
    if relsize <= 1:
        new_shape = [int(round(res * relsize))
                     for res in image.shape]

        if avoidodds:
            new_shape = tuple([int(res + res % 2)
                               for res in new_shape])

        image = imresize(image, new_shape, 'bicubic')
        image = np.float32(image)
    return image


def get_logpolar(array, interpolation=0, reverse=False):
    """
    Returns a new array with the logpolar transfamation of array.
    Interpolation can be:
        0 Near
        1 Linear
        2 Bilineal
        3 Cubic
        4
        5
    """
    assert interpolation in range(6)
    rows, cols = array.shape
    row0 = rows / 2.
    col0 = cols / 2.
    theta_scalar = tau / cols
    max_radius = (row0 ** 2 + col0 ** 2) ** .5
    rho_scalar = log(max_radius) / cols

    def cart2logpol(dst_coords):
        theta, rho = dst_coords
        rho = exp(rho * rho_scalar)
        theta = np.pi / 2 - theta * theta_scalar
        row_from = rho * cos(theta) + row0
        col_from = rho * sin(theta) + col0
        return row_from, col_from

    def logpol2cart(dst_coords):
        xindex, yindex = dst_coords
        x = xindex - col0
        y = yindex - row0

        r = np.log(np.sqrt(x ** 2 + y ** 2)) / rho_scalar
        theta = np.arctan2(y, x)
        theta_index = np.round((theta + np.pi) * cols / tau)
        return theta_index, r

    trans = logpol2cart if reverse else cart2logpol

    logpolar = geometric_transform(array, trans, array.shape,
                                   order=interpolation)

    return logpolar


def get_polar(array, interpolation=0, reverse=False):
    """
    Returns a new array with the logpolar transfamation of array.
    Interpolation can be:
        0 Near
        1 Linear
        2 Bilineal
        3 Cubic
        4
        5
    """
    assert interpolation in range(6)
    rows, cols = array.shape
    row0 = rows / 2.
    col0 = cols / 2.
    theta_scalar = tau / cols
    max_radius = (row0 ** 2 + col0 ** 2) ** .5
    rho_scalar = max_radius / cols

    def cart2pol(dst_coords):
        theta, rho = dst_coords
        rho = rho * rho_scalar
        theta = np.pi / 2 - theta * theta_scalar
        row_from = rho * cos(theta) + row0
        col_from = rho * sin(theta) + col0
        return row_from, col_from

    def pol2cart(dst_coords):
        xindex, yindex = dst_coords
        x = xindex - col0
        y = yindex - row0

        r = np.sqrt(x ** 2 + y ** 2) / rho_scalar
        theta = np.arctan2(y, x)
        theta_index = np.round((theta + np.pi) * cols / tau)
        return theta_index, r

    trans = pol2cart if reverse else cart2pol

    polar = geometric_transform(array, trans, array.shape,
                                order=interpolation)

    return polar


def open_raw(filename):
    known_resolutions = {
        5038848: (1944, 2592, "bayer"),
        262144: (512, 512, "mono"),
        266638: (512, 520, "mono"),
    }

    bits = open(filename, "rb").read()
    lenght = len(bits)

    if lenght in known_resolutions:
        rows, cols, method = known_resolutions[lenght]
        array = np.array([ord(char) for char in bits])
        array = array.reshape((rows, cols))

        if method == "bayer":
            # TODO: implement Malvar-He-Cutler Bayer demosaicing
            print("Identified %s as bayer raw." % filename)
            array0 = array[0::2, 0::2]
            array1 = array[0::2, 1::2]
            array2 = array[1::2, 0::2]
            array3 = array[1::2, 1::2]
            red = array1
            green = (array0 + array3) / 2
            blue = array2
            array = np.array([red, green, blue])

        return array

    else:
        raise IOError(f"unknown resolution on raw file {filename} "
                      f"({lenght:d} pixels)")


def imread(filename, flatten=True):
    if filename.endswith(".raw"):
        array = open_raw(filename)
    else:
        array = misc.imread(filename, flatten)
    return array


def evenshape(array, shrink=False):
    if not shrink:
        newshape = [dim + 1 - dim % 2 for dim in array.shape]
        newarray = np.zeros(newshape)
        newarray[:array.shape[0], :array.shape[1]] = array
    else:
        newshape = [dim - 1 + dim % 2 for dim in array.shape]
        newarray = array[:newshape[0], :newshape[1]]
    return newarray


def imwrite(array, filename):
    return imsave(filename, array)


def get_centered(array, center=None, mode='wrap', reverse=False):
    """
    Shift the given array to make the given point be the new center.
    If center is None the center of mass is used.
    mode can be 'constant', 'nearest', 'reflect' or 'wrap'.

    inverse False:  center -> current_center
    inverse True:   current_center -> center
    """

    if center:
        rows, cols = array.shape
        rowcc = int(round(rows / 2.))
        colcc = int(round(cols / 2.))
        rowc, colc = center
        if reverse:
            drows = rowc - rowcc
            dcols = colc - colcc
        else:
            drows = rowcc - rowc
            dcols = colcc - colc
        shift = (drows, dcols)
    else:
        if issubclass(array.dtype.type, complex):
            intensity = get_intensity(array)
            shift = get_shift_to_center_of_mass(intensity, mode)
        else:
            shift = get_shift_to_center_of_mass(array, mode)

    if issubclass(array.dtype.type, complex):
        real = ndimage.shift(array.real, shift, mode=mode)
        imag = ndimage.shift(array.imag, shift, mode=mode)
        centered = real + 1j * imag
    else:
        centered = ndimage.shift(array, shift, mode=mode)

    return centered


@cache.hybrid
def get_shift_to_center_of_mass(array, mode="wrap"):
    """
    Calcules the shift of the center of mass relative to the center of the image
    """
    if array.ndim > 1:
        shift = [get_shift_to_center_of_mass(array.sum(dim))
                 for dim in range(array.ndim)][::-1]
        return shift
    else:
        center = array.shape[0] / 2.
        total_shift = 0
        centered = array
        for step in range(100):
            center_of_mass = ndimage.center_of_mass(centered)
            shift = center - center_of_mass[0]
            eshift = shift * 2 ** .5
            if abs(eshift) < 1:
                break
            total_shift += eshift
            centered = ndimage.shift(centered, eshift, mode=mode)

        shift = int(round(total_shift))

        return shift


def get_intensity(array):
    return array.real ** 2 + array.imag ** 2


def logscale(array):
    array = array.copy()
    if issubclass(array.dtype.type, complex):
        array = get_intensity(array)
    array = array.astype(float)
    array -= array.min()
    array *= np.expm1(1) / array.max()
    array = np.log1p(array)
    array *= 255.
    return array


def normalize(array):
    """
    Apply linears tranformations to ensure all the values are in [0, 255]
    """
    array = array.copy()
    if issubclass(array.dtype.type, complex):
        array = get_intensity(array)
    array -= array.min()
    array *= 255. / array.max()
    return array


def equalizearray(array):
    """
    Equalize the array histogram
    """
    array = normalize(array)
    array[array < 10e-10] = 0                    # enough precision
    if issubclass(array.dtype.type, complex):
        array = get_intensity(array)
    array = array.astype(float)
    shape = array.shape
    array = array.flatten()
    sorters = array.argsort()
    array.sort()
    zippeds = zip(array, sorters)
    groups = itertools.groupby(zippeds, operator.itemgetter(0))
    counter = itertools.count()
    for ovalue, group in groups:
        value = counter.next()
        for ovalue, pos in list(group):
            array[pos] = value
    if value:
        array *= 255. / value
    array = array.reshape(shape)
    return array


def equalize(image):
    if isinstance(image, pil.Image):
        if image.mode in ("F"):
            return equalizearray(np.asarray(image))
        elif image.mode in ("RBGA"):
            image = image.convert("RBG")
        return ImageOps.equalize(image)
    else:
        return equalizearray(image)


def example_clock(side=250, time=None):
    """
    Just kept here for coding documentation of how to efficintly use these
    helper functions.

    Time may be a tuple (hour, minute, second), if not current time is used.
    """

    if time is None:
        from datetime import datetime
        now = datetime.now()
        hour, minute, second = now.hour, now.minute, now.second
    else:
        hour, minute, second = time

    img = np.ones((side, side))
    hours = Frame(img, pins=12)
    minutes = Frame(img, pins=60)
    seconds = Frame(img, pins=60)
    center = ((side - 1) // 2, (side - 1) // 2)
    img[center] = 0

    draw_line(img, center, hours[hour],     value=0, opacity=1)
    draw_line(img, center, minutes[minute], value=0, opacity=.5)
    draw_line(img, center, seconds[second], value=0, opacity=.25)

    imshow(img, f"Clock arrows pointing {hour}:{minute:02d}:{second:02d}")
