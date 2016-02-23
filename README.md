Make this have words
#LMFM

A factorisation machine written in cython, trained using Alternating Least Squares and sklearn compatible!

#Installation
```pip install lmfm```
Done!
Requires cython, numpy, scipy and sklearn.

#Example
```
from sklearn.datasets import load_boston
from lmfm import LMFM
from sklearn.cross_validation import cross_val_predict
d = load_boston()
X = d.data
y = d.target

fm = LMFM(n_iter=100, verbose=0)

preds = cross_val_predict(fm, X, y)
```
