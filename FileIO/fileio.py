import os
import deepdish

base_dir = 'files'


def extend_path(path):
    return os.path.join(base_dir, path)
    
    
def save(obj, path):
    deepdish.io.save(obj, extend_path(path))


def load(path):
    return deepdish.io.load(extend_path(path))


def check(path):
    return os.path.exists(extend_path(path))


def remove(path):
    os.remove(extend_path(path))
