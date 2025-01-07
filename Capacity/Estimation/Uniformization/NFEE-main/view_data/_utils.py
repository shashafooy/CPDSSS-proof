import os
import sys


def set_sys_path():
    root_path = os.path.abspath(os.path.dirname(__file__))
    while os.path.basename(root_path) != "NFEE-main":
        root_path = os.path.dirname(root_path)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
