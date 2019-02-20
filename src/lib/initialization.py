"""
Set of functions to quickly estimate weights.
"""


def direct_iterative_weights(target, pins):
    """
    For each possible thread minimices naive fitness.
    This is: ((1-thread) * target).sum()
    """
    return weights


def optimize_weights(target, pins, initial_weights=None):
    """
    Uses Newton pan-dimensional optimization to calculate/improve the weights.
    This improves the output since considers the interaction between threads.
    """
    return weights
