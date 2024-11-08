import numpy as np


def integer_factors(a):
    """
    Finds the largest integer factors of a. a = f1 * f2, where f1 >= f2.
    If there are no non-trivial solutions, returns sqrt(a) + 1
    :param a: integer
    :return: f1, f2
    """
    # the bigger factor is between a/2 and sqrt(a)
    upper = int(a / 2)
    lower = int(np.ceil(np.sqrt(a)))  # otherwise for e.g. 20 the factors are (4,5) instead of (5,4) ;)
    # check each combination
    for f1 in range(lower, upper + 1):
        f2 = a / f1
        # return if found a solution
        if f2.is_integer():
            return int(f1), int(f2)
    # since i want this for plotting...
    # return int(sqrt(a) + 1) instead
    if lower * (lower-1) >= a:
        return lower, int(lower-1)
    else:
        return lower, lower
