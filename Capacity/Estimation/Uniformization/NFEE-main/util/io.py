# import cPickle as pickle
import pickle
import os
import sys
import numpy as np
import fcntl


def save(data, file):
    """
    Saves data to a file.
    """

    dir = os.path.dirname(file)
    if dir:
        make_folder(dir)

    with open(file + ".pkl", "wb") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        pickle.dump(data, f)
        fcntl.flock(f, fcntl.LOCK_UN)


def load(file):
    """
    Loads data from file.
    """

    with open(file + ".pkl", "rb") as f:
        data = pickle.load(f)

    if hasattr(data, "reset_theano_functions"):
        data.reset_theano_functions()

    return data


def save_txt(str, file):
    """
    Saves string to a text file.
    """

    dir = os.path.dirname(file)
    if dir:
        make_folder(dir)

    with open(file, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(str)
        fcntl.flock(f, fcntl.LOCK_UN)


def load_txt(file):
    """
    Loads string from text file.
    """

    with open(file, "r") as f:
        str = f.read()

    return str


def save_txt_from_numpy(data, file):
    dir = os.path.dirname(file)
    if dir:
        make_folder(dir)

    np.savetxt(file, data, fmt="%f", delimiter=" ")


def make_folder(folder):
    """
    Creates given folder (or path) if it doesn't exist.
    """

    if not os.path.exists(folder):
        os.makedirs(folder)


class Logger:
    """
    Implements an object that logs messages to a file, as well as printing them on the sceen.
    """

    def __init__(self, filename):
        """
        :param filename: file to be created for logging
        """
        self.f = open(filename, "w")

    def write(self, msg):
        """
        :param msg: string to be logged and printed on screen
        """
        sys.stdout.write(msg)
        self.f.write(msg)

    def __enter__(self):
        """
        Context management enter function.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context management exit function. Closes the file.
        """
        self.f.close()
        return False
