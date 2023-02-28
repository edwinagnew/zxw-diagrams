from discopy.quantum.zx import Spider, Id
from discopy import tensor, Tensor, Dim
import numpy as np
import sys

from discopy.quantum.zx import Functor
import tensornetwork as tn

np.set_printoptions(threshold=sys.maxsize)


class Z(Spider):
    """ Z spider. """
    def __init__(self, n_legs_in, n_legs_out, phase=0):
        super().__init__(n_legs_in, n_legs_out, phase, name='Z')
        self.color = "green"

    @property
    def array(self):
        n, m = len(self.dom), len(self.cod)
        array = np.zeros(1 << (n + m), dtype=complex)
        array[0] = 1
        array[-1] = np.exp(1j * self.phase)
        return Tensor(Dim(2) ** n, Dim(2) ** m, array)
    
    
def one_hots(n):
    zeros = '0'*n
    strings = []
    for i in range(n):
        strings.append('0'*i + '1' + '0'*(n-i-1))
    return strings

class W(Spider):
    def __init__(self, n=2, mon=True):
        if mon:
            super().__init__(1, n) # assume 1 in for now 
            self.color = "black" 
            self.shape = "triangle_up"
        else:
            super().__init__(n, 1)
            self.color = "black" 
            self.shape = "triangle_down"
            
        self.mon = mon
        self.n = n
            
        
        
    @property
    def array(self):
        # |0..0><0| + (|10..> + |01..> + ...)<1|
        n = len(self.cod)
        
        array = np.zeros(2 ** (1 + n))
        array[0] = 1.0
        for j in one_hots(n):

            array[2**n + int(j, 2)] = 1
            
        if self.mon:
            return Tensor(Dim(2), Dim(2)**n, array)
        else:
            array[2 * int(j, 2) + 1] = 1 !!
            return Tensor(Dim(2)**n, Dim(2), array)
    
    def T(self):
        return type(self)(n=self.n, mon=not self.mon)
        
    
    


def f_ob(ob):
    return Dim(2) ** len(ob)

def f_ar(box):
    return tensor.Box(box.name, f_ob(box.dom), f_ob(box.cod), box.array)


def eval(diagram):
    d = Functor(ob=f_ob, ar=f_ar, ar_factory=tensor.Diagram)(diagram)
    t = d.eval(contractor=tn.contractors.auto)
    
    n, m = len(diagram.dom), len(diagram.cod)
    #print(n, m, "\n")
    
    return t.array.astype(float).reshape(2**n, 2**m).transpose()
    #return t.array.astype(float).reshape(dim, dim).transpose()