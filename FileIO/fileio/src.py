import os
import deepdish

base_dir = 'files'


def _assert_dir_exists():
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)


def _extend_path(path):
    return os.path.join(base_dir, path)


def save(obj, path):
    _assert_dir_exists()
    deepdish.io.save(_extend_path(path), obj)


def load(path):
    return deepdish.io.load(_extend_path(path))


def check(path):
    return os.path.exists(_extend_path(path))


def remove(path):
    os.remove(_extend_path(path))
