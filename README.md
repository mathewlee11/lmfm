#LMFM

A factorisation machine written in cython, trained using Alternating Least Squares and sklearn compatible!

#Installation
```pip install lmfm```

Done! You can also get the dev version with ```pip install git+git://github.com/mathewlee11/lmfm```
Requires cython, numpy, scipy and sklearn.

#Example
```
from sklearn.datasets import load_boston
from lmfm import LMFMRegressor
from sklearn.cross_validation import cross_val_predict
d = load_boston()
X = d.data
y = d.target

fm = LMFMRegressor(n_iter=100)

preds = cross_val_predict(fm, X, y)
```
