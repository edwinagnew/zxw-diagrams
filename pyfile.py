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


def eval(diagram, round=True):
    d = Functor(ob=f_ob, ar=f_ar, ar_factory=tensor.Diagram)(diagram)
    t = d.eval(contractor=tn.contractors.auto)
    
    n, m = len(diagram.dom), len(diagram.cod)
    #print(n, m, "\n")
    
    if round:
        #return np.round(t.array, 5).astype(complex).reshape(2**n, 2**m).transpose()
        return np.round(t.array, 5).reshape(2**m, 2**n)
    else:
        return t.array.reshape(2**m, 2**n)
    
    
def eq(a, b, close=True):
    return np.allclose(eval(a, round=False), eval(b, round=False))
    

