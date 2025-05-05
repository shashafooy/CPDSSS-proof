import re
import numpy as np
import shutil
from scipy.optimize import lsq_linear


def remove_whitespace(str):
    """
    Returns the string str with all whitespace removed.
    """

    p = re.compile(r"\s+")
    return p.sub("", str)


def prepare_cond_input(xy, dtype):
    """
    Prepares the conditional input for model evaluation.
    :param xy: tuple (x, y) for evaluating p(y|x)
    :param dtype: data type
    :return: prepared x, y and flag whether single datapoint input
    """

    x, y = xy
    x = np.asarray(x, dtype=dtype)
    y = np.asarray(y, dtype=dtype)

    one_datapoint = False

    if x.ndim == 1:

        if y.ndim == 1:
            x = x[np.newaxis, :]
            y = y[np.newaxis, :]
            one_datapoint = True

        else:
            x = np.tile(x, [y.shape[0], 1])

    else:

        if y.ndim == 1:
            y = np.tile(y, [x.shape[0], 1])

        else:
            assert x.shape[0] == y.shape[0], "wrong sizes"

    return x, y, one_datapoint


def prepare_cond_input_ed(xdy, dtype):
    """
    Prepares the conditional input for model evaluation.
    :param xdy: tuple (x, d, y) for evaluating p(y|x, d)
    :param dtype: data type
    :return: prepared x, d, y and flag whether single datapoint input
    """

    x, d, y = xdy
    x = np.asarray(x, dtype=dtype)
    d = np.asarray(d, dtype=dtype)
    y = np.asarray(y, dtype=dtype)

    one_datapoint = False

    if x.ndim == 1:

        assert d.ndim == 1, "wrong sizes"

        if y.ndim == 1:
            x = x[np.newaxis, :]
            d = d[np.newaxis, :]
            y = y[np.newaxis, :]
            one_datapoint = True

        else:
            x = np.tile(x, [y.shape[0], 1])
            d = np.tile(d, [y.shape[0], 1])

    else:

        if d.ndim == 1:
            d = np.tile(d, [x.shape[0], 1])

        else:
            assert x.shape[0] == d.shape[0], "wrong sizes"

        if y.ndim == 1:
            y = np.tile(y, [x.shape[0], 1])

        else:
            assert x.shape[0] == y.shape[0], "wrong sizes"

    return x, d, y, one_datapoint


# Print iterations progress
def printProgressBar(
    iteration, total, prefix="", suffix="", decimals=1, length=70, fill="█", printEnd="\r"
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    terminal_width = shutil.get_terminal_size().columns
    reserved = len(prefix) + len(suffix) + len("100.%") + decimals + 7
    length = min(length, terminal_width - reserved)

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def split_complex_to_real(arr):
    real = np.real(arr)
    imag = np.imag(arr)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]  # add dimension to act like matrix
    n_samp, N = arr.shape
    output = np.empty((n_samp, 2 * N), dtype=real.dtype)
    output[:, 0::2] = real
    output[:, 1::2] = imag

    if output.shape[0] == 1:
        return output[0]  # arr was originally a vector
    else:
        return output


def lsq_single(args):
    A, b, bounds = args
    return lsq_linear(A, -b, bounds=bounds).x
