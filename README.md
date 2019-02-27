# Kaggle Utils
Here, I will store my utils scripts that helped me in the Kaggle competitions

## AutoEnsembler

AutoEnsembler is a simple tool that allows to achieve +2%-3% validation accuracy during models ensembling. 
It works by smartly splitting the validation set, ierarchically selecting best blending parameters and by consequently shrinking the grid size. It has `n_jobs` parameter to allow for faster computations.

AutoEnsembler was used in these competitions:
* [Humpback Whales Identification](https://www.kaggle.com/c/humpback-whale-identification): 0.6% LB improvement, 2% CV improvement compared to linear blending.


## FileIO

FileIO is a simple library that allows you to store numerical objects in a fast and efficient way.

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
