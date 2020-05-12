from __future__ import print_function
import numpy as np
import theano.tensor as T
from theano import function

# basic
x = T.dscalar('x')
y = T.dscalar('y')
z = x+y
f = function([x, y], z)

 # only give the inputs "x and y" for this function, then it will calculate the output "z"
print(f(2,3))

# to pretty-print the function
from theano import pp
print(pp(z))

# how about matrix
x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x, y], z)
print(f(np.arange(12).reshape((3,4)), 10*np.ones((3,4))))
