#!/usr/bin/env python
#-*- coding: UTF-8 -*-

from scipy import optimize, stats
from numpy import pi

import numpy as np

tau = 2 * pi


def generic_minimizer(fitness_func, initial_guess, optimizers=None):
    """
    A common interface to various minimization algorithms
    """

    if optimizers == None:
        # TODO: Create a dinamic fitter for best know solutions.
        optimizers = [
            optimize.fmin, # 66
#             optimize.fmin_powell,
#            optimize.leastsq,
        ]

    best_result = None
    for optimizer in optimizers:
        xend = optimizer(fitness_func, initial_guess, disp=False)
        last_result = fitness_func(xend)
        if best_result is None or last_result < best_result:
            best_guess = xend
            best_result = last_result

    return best_guess


def get_paraboloid(x, y, a0, b0, a1, b1, c=0):
    """
    Perfect model for lens distorsion.

    a0 * (x - b0) ** 2 + a1 * (y - b1) ** 2 + c
    """
    return a0 * (x - b0) ** 2. + a1 * (y - b1) ** 2 + c


def wrapped_gradient(phase):
    rows, cols = phase.shape
    dx, dy = np.gradient(phase)
    for diff in (dx, dy):
        diff[diff < -pi / 2] += pi
        diff[diff > pi / 2] -= pi

    return dx, dy


def get_fitted_paraboloid(data):
    """
    Adjust a paraboloid to the input data using normal linear regression over
    the gradient of each dimension outline.
    This method allow us to correct a wrapped phase paraboloic deformation.
    """
    xs, ys = data.shape
    x = np.mgrid[:xs]
    y = np.mgrid[:ys]

    diff_x, diff_y = wrapped_gradient(data)
    diff_outline_x = diff_x.mean(1)
    diff_outline_y = diff_y.mean(0)

    dax, dbx, r_value, p_value, std_err = stats.linregress(x, diff_outline_x)
    day, dby, r_value, p_value, std_err = stats.linregress(y, diff_outline_y)

    ax = dax / 2 
    bx = - dbx / dax
    ay = day / 2 
    by = - dby / day
    x, y = np.mgrid[:xs, :ys]
    return get_paraboloid(x, y, ax, bx, ay, by)


def example_lens():
    """
    Just a refresher on arrays manipulation and representation
    """

    from skimage.data import astronaut
    from .interface import imshow

    x, y = np.mgrid[:512, :512]
    eye = astronaut().mean(2) # value
    data = get_paraboloid(x, y, 1, 250, 3, 300, 5)
    data /= data.ptp() / 256. * 0.25
    noisy = (data + eye).astype(float)
    noisy /= noisy.ptp() / 20
    noisy %= tau
    fitted = get_fitted_paraboloid(noisy)

    imshow(eye, "Original Image")
    imshow(noisy, "With lens distorsion added")
    imshow(fitted % tau, "Detected distorsion")
    imshow((noisy - fitted) % tau, "Fixed image")

    return 0
