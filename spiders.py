from discopy.quantum.zx import Spider, Id, Box
from discopy import tensor, Tensor, Dim
from discopy.rigid import PRO
import numpy as np
import sys

from pyfile import eval


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
        array[0] = 1.0
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
        n = self.n
        
        array = np.zeros(2 ** (1 + n))
        array[0] = 1.0
        for j in one_hots(n):

            if self.mon:
                array[2**n + int(j, 2)] = 1
            else:
                array[2 * int(j, 2) + 1] = 1
            
        if self.mon:
            return Tensor(Dim(2), Dim(2)**n, array)
        else:
            return Tensor(Dim(2)**n, Dim(2), array)
    
    def T(self):
        return type(self)(n=self.n, mon=not self.mon)
    
    

H = Box('H', PRO(1), PRO(1))
H.dagger = lambda: H
H.draw_as_spider = True
H.drawing_name, H.tikzstyle_name, = '', 'H'
H.color, H.shape = "yellow", "rectangle"
H.array = Tensor(Dim(2), Dim(2), 1/np.sqrt(1) * np.array([1.0, 1, 1, -1]))

CZ = Z(1, 2) @ Id(1) >> Id(1) @ H @ Id(1) >> Id(1) @ Z(2, 1)

Swap = Id(2).swap(1, 1)

FSwap = Box('O', PRO(2), PRO(2), data=eval(Swap >> CZ).flatten(), draw_as_spider=True)
FSwap.array = FSwap.data
FSwap.color = 'white'
