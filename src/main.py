import time
import random
import itertools
from functools import reduce
from operator import mul

from scipy import optimize
import IPython
import click
import numpy as np

from .lib import image, interface, minimize, fitness
from .lib.interface import progressbar


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])  # should be default


class Echo:

    def __init__(self, every=10):
        self.every = every
        self.count = 0

    def __call__(self, string):
        if not (self.count % self.every):
            click.echo("%s  (%d)" % (string, self.count))
        self.count += 1

echo_100 = Echo(every=100)
echo_1000 = Echo(every=1000)

@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """
    Putting all together
    """
    click.echo(click.style("Let's roll!", fg="red"))


def load_target_image(target, max_side=200):
    img_target = image.imread(target)
    rows, cols = img_target.shape
    click.echo(f"{target} [{img_target.shape}]")
    if rows != cols:
        limit = min(rows, cols)
        diff = max(rows, cols) - limit
        margin = diff // 2
        if rows > limit:
            img_target = img_target[margin:limit + margin, :]
        else:
            img_target = img_target[:, margin:limit + margin]
        click.echo(f" -> re-dim to [{img_target.shape}]")
    shape = img_target.shape
    img_target = image.limit_size(img_target, max_side ** 2)
    img_target /= img_target.max()  # domain (0, 1]
    if shape != img_target.shape:
        click.echo(f" -> reduced to [{img_target.shape}]")
    return img_target

@cli.command()
@click.argument('target', type=click.Path(exists=True, dir_okay=False))
@click.option('--depth', default=3, type=click.IntRange(1, 100),
              help='The number of recursive threads.')
@click.option('--pins', default=120, type=click.IntRange(3, 500),
              help='How many pins there are in the frame.')
@click.option('--max-side', default=1000, type=click.IntRange(100, 5000),
              help='Limit resolution of images.')
def knit(target, depth, pins, max_side):
    img_target = load_target_image(target, max_side)

    frame = image.Frame(img_target, pins=pins)

    img_target_mask = np.ones_like(img_target)
    frame_mask = frame.get_mask()
    img_target_mask[frame_mask] = img_target[frame_mask]
    img_target = img_target_mask
#     image.imshow(img_target)
    print(img_target.ptp())

    initial_guess = []
    with progressbar(frame.segments(), label="Segments shadow") as items:
        for pin_from, pin_to in items:
            rows, cols = frame.get_segment_pixels(pin_from, pin_to)
            initial_guess.append(img_target[rows, cols].sum())
    initial_guess = np.array(initial_guess)
    initial_guess /= initial_guess.max()
    initial_guess = initial_guess ** 2
    click.echo(f"{initial_guess.min()} - {initial_guess.mean()}"
               f"- {initial_guess.max()}")

#     img_initial_guess = frame.render(initial_guess)
#     img_initial_guess_fast = frame.render(initial_guess, fast=True)
#     image.imshow(img_initial_guess, "Normal render")
#     image.imshow(img_initial_guess_fast, "Fast render")

    img_target_blur = image.blur(img_target, 2)

    def fitness_func(weights):
        img_render = frame.render(weights, fast=True, blur_sigma=2)
        diff = abs(fitness.default(img_target_blur, img_render))
        click.echo(diff / max_side ** 2)
        return diff

    best_guess = optimize.minimize(fitness_func, initial_guess,
            options={"maxiter": 10, "disp": True})
    IPython.embed()
    img_best_guess = frame.render(best_guess)
    image.imshow(img_best_guess)

