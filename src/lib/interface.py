#!/usr/bin/env python


from functools import partial
from io import StringIO
import os
import sys

import click
import numpy as np

progressbar = partial(click.progressbar, label='Progress',
                      bar_template='%(label)s  %(bar)s | %(info)s',
                      fill_char=click.style(u'█', fg='cyan'), empty_char=' ',
                      show_eta=True)

if os.name in ("nt") or "TKPIPE" in os.environ:
    import tkpipe
    import Image as pil
    TKPIPE = tkpipe.Tkpipe()
    sys.stdout = TKPIPE.default("green")
    sys.stderr = TKPIPE.default("red")
else:
    from matplotlib import pyplot as plt
    TKPIPE = False


def fig2raster(figure):
    """
    Convert a matplotlib to a raster PIL image
    """
    if hasattr(figure, "savefig"):
        fileo = StringIO()
        figure.savefig(fileo)
        fileo.seek(0)
        figure = pil.open(fileo)
    return figure


def imshow(image, title=None):
    image = fig2raster(image)
    if TKPIPE:
        raise NotImplementedError("Must print title!")
        if isinstance(image, np.ndarray):
            try:
                image = pil.fromarray(image)
            except TypeError:
                image = pil.fromarray(np.float64(image))
            except IndexError:
                print(image)
                raise
        TKPIPE.writeimage(image)
    else:
        try:
            plot = plt.imshow(image)
        except TypeError:
            plot = plt.imshow(image.astype(float))
        except ValueError:
            if type(image) is tuple:
                plot = plt.imshow(image[0])
            else:
                raise

        plot.set_cmap('gray')
        plot.axes.set_title(title)
        plot.axes.set_axis_off()
        plot.axes.margins(0)
        return plt.show()


def color(message, color="blue"):
    if TKPIPE:
        TKPIPE.write(message, color)
    else:
        sys.stderr.write(message)
    return message


def example_progress(count):
    """Demonstrates the progress bar."""
    import time
    import random

    items = range(count)

    def process_slowly(item):
        time.sleep(0.002 * random.random())

    with progressbar(items) as b_items:
        for item in b_items:
            process_slowly(item)

    with progressbar(item for item in items) as b_items:
        for item in b_items:
            process_slowly(item)
