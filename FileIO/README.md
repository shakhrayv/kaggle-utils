## FileIO

FileIO is a simple library that allows you to store numerical objects in a fast and efficient way.

**Installation:**

```
$ git clone https://github.com/codelovin/kaggle-utils.git
$ cd kaggle-utils/FileIO
$ python3 setup.py install
```

**Usage:**

```
import fileio

array = {
    'numpy': numpy.array([1, 2]),
    'string': 'fileio!'
}

fileio.save(array, 'example')

assert fileio.check('example')
assert fileio.load('example')['string'] == 'fileio!'

fileio.remove('example')
```
