from discopy import tensor, Tensor, Dim
import numpy as np

from discopy.quantum.zx import Functor
import tensornetwork as tn

import sys

np.set_printoptions(threshold=sys.maxsize)





def f_ob(ob):
    return Dim(2) ** len(ob)

def f_ar(box):
    return tensor.Box(box.name, f_ob(box.dom), f_ob(box.cod), box.array)


def eval(diagram):
    d = Functor(ob=f_ob, ar=f_ar, ar_factory=tensor.Diagram)(diagram)
    t = d.eval(contractor=tn.contractors.auto)
    
    n, m = len(diagram.dom), len(diagram.cod)
    #print(n, m, "\n")
    
    return t.array.astype(complex).reshape(2**n, 2**m).transpose()
    #return t.array.astype(float).reshape(dim, dim).transpose()
    
    
def eq(a, b, close=True):
    return np.allclose(eval(a), eval(b))
    

