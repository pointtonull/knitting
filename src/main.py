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

def fitness_func(weights, frame, img_target):
    img_render = frame.render_weights(weights, fast=True, blur_sigma=2)
    pixel_diff = ((img_target - img_render) ** 2).avg()
    echo_1000("Px Diff: %f" % pixel_diff)
    return pixel_diff

def differential_evolution_strategies():
    bounds = [(0, 1)] * ((pins ** 2 - pins) // 2)
    min_strategy = (np.inf, None)
    for strategy in ["best1bin", "best1exp", "rand1exp", "randtobest1exp",
                     "currenttobest1exp", "best2exp", "rand2exp",
                     "randtobest1bin", "currenttobest1bin", "best2bin",
                     "rand2bin", "rand1bin"]:
        best_guess = optimize.differential_evolution(fitness_func, bounds,
                                                     args=(frame,
                                                           img_target_blur),
                                                     disp=True,
                                                     popsize=1,
                                                     strategy=strategy,
                                                     maxiter=2,
                                                     updating="deferred",
                                                     workers=4,
                                                     )
    print(strategy)
    print(best_guess["func"])


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
    img_target_blur = image.blur(img_target, 2)

    min_cost_segment = (np.inf, None)
    n_segments = (pins ** 2 - pins) // 2
    with progressbar(frame.segments(), length=n_segments,
                     label="Initialize canvas") as segments:
        for segment in segments:
            cost = fitness.naive(img_target_blur, frame.render_segment(segment))
            min_cost_segment = min(min_cost_segment, (cost, segment))

    cost, steps = min_cost_segment
    click.echo(f"[{steps}] Initial Cost: {cost}")
    steps = list(steps)
    img_render = frame.render_segment(min_cost_segment[1], fast=True,
                                      blur_sigma=0)
    min_cost_steps = (np.inf, None)
    improvement = 1
    for iteration in itertools.count(1):
        for new_steps in itertools.combinations(range(frame.pins), depth):
            last_step = steps[-1]
            if last_step in new_steps:
                continue
            segments = list(zip([last_step] + list(new_steps[:-1]), new_steps))
            img_new_render = img_render.copy()
            rows, cols = reduce(lambda prev, new: ((np.append(prev[0], new[0]),
                                                   (np.append(prev[1], new[1])))),
                                (frame.get_segment_pixels(*seg)
                                 for seg in segments))
            img_new_render[rows, cols] *= .5
            new_cost = fitness.naive(img_target_blur, img_new_render)
            min_cost_steps = min(min_cost_steps, (new_cost, new_steps))

        new_cost, new_steps = min_cost_steps
        steps.append(new_steps[0])
        img_render *= frame.render_segment(tuple(steps[-2:]))
        improvement = cost - new_cost
        cost = new_cost
        click.echo(f"[{iteration:3d}] [{steps[-1]:3d}] "
                   f"Cost: {cost:.6} (-{improvement:.2e})")
        if improvement <= 0:
            steps.extend(new_steps[:-1])
            break

#         if not(iteration % 10):
#             image.imshow(img_render, "%s" % steps[-10:])

    image.imshow(img_render, "Final result")
    IPython.embed()
    return
