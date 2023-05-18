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
        
        #assert -2 <= phase <= 2, "phase should be multiple of pi"

    @property
    def array(self):
        n, m = len(self.dom), len(self.cod)
        array = np.zeros(1 << (n + m), dtype=complex)
        array[0] = 1.0
        array[-1] = np.exp(1j * self.phase)
        return Tensor(Dim(2) ** n, Dim(2) ** m, array)
    
    
class ZBox(Spider):
    """ Green box. """
    def __init__(self, n_legs_in, n_legs_out, phase=0):
        super().__init__(n_legs_in, n_legs_out, phase, name='ZBox')
        self.color = "green"
        self.shape = 'rectangle'
        

    @property
    def array(self):
        n, m = len(self.dom), len(self.cod)
        array = np.zeros(1 << (n + m), dtype=complex)
        array[0] = 1
        array[-1] = self.phase
        return Tensor(Dim(2) ** n, Dim(2) ** m, array)
    
def boxes(ps, states=True):
    if len(ps) == 1:
        return ZBox(int(not states), 1, ps[0])
    return ZBox(int(not states), 1, ps[0]) @ boxes(ps[1:], states=states)
    
def bitstring(x):
    return [int(b) for b in "{0:b}".format(x)]
    
class X(Spider):
    """ X spider. """
    def __init__(self, n_legs_in, n_legs_out, phase=0):
        super().__init__(n_legs_in, n_legs_out, float(phase), name='X')
        self.color = "red"

    @property
    def array(self):
        
        assert self.phase in (0, 1.0)
        n, m = len(self.dom), len(self.cod)
        array = np.zeros(1 << (n + m), dtype=complex)
        bit = 1 if self.phase == 1.0 else 0
        for i in range(len(array)):
            parity = (bitstring(i).count(1) + bit) % 2
            array[i] = 1 - parity
        return Tensor(Dim(2) ** n, Dim(2) ** m, array)
       
    
    
def one_hots(n): # could have just used powers of 2 lol
    zeros = '0'*n
    strings = []
    for i in range(n):
        strings.append('0'*i + '1' + '0'*(n-i-1))
    return strings

def w_mat(m, n):
    mat = np.zeros((2**m, 2**n)) # rows, columns
    for i in range(2**n - 1):
        bi = format(i, '0' + str(n) + 'b')
        if bi.count('0') == 1:
            mat[0][i] = 1.0

    for j in range(m): # all powers of 2 in final column
        mat[2**j][-1] = 1.0

    return mat

class W_old(Spider):
    def __init__(self, n=2, mon=True, norm=False):
        self.norm_factor = 1.0 if not norm else 1/np.sqrt(n)
        
        if mon:
            super().__init__(1, n, 0, name='W') # assume 1 in for now 
            self.color = "black" 
            self.shape = "triangle_up"
        else:
            super().__init__(n, 1, 0, name='W')
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
                array[2**n + int(j, 2)] = 1 * self.norm_factor
            else:
                array[2 * int(j, 2) + 1] = 1 * self.norm_factor
            
        if self.mon:
            return Tensor(Dim(2), Dim(2)**n, array)
        else:
            return Tensor(Dim(2)**n, Dim(2), array)
    
  
    def dagger(self):
        return type(self)(n=self.n, mon=not self.mon, norm = self.norm_factor != 1)
    
class W(Spider):
    def __init__(self, n=1, m=2, down=False, norm=False):
        self.norm_factor = 1.0 if not norm else np.sqrt(n)/np.sqrt(m)
        
        #assert not down, "havent worked out tranpose yet"
        
        super().__init__(n, m, 0, name='W')
        self.color = "black"
        self.shape = "triangle_down" if down else "triangle_up"
        
        self.down = down
        self.n = n
        self.m = m
        
    @property
    def array(self):
        if not self.down:
           
            mat = w_mat(self.m, self.n) * self.norm_factor
            return Tensor(Dim(2)**self.n, Dim(2)**self.m, mat.transpose())
        
        else: # flip (and then tranpose later)
            mat = w_mat(self.n, self.m) * self.norm_factor
            return Tensor(Dim(2)**self.n, Dim(2)**self.m, mat)
        
        
           
    def dagger(self):
        return type(self)(n=self.m, m=self.n, down=not self.down)
    
    

H = Box('H', PRO(1), PRO(1))
H.dagger = lambda: H
H.draw_as_spider = True
H.drawing_name, H.tikzstyle_name, = '', 'H'
H.color, H.shape = "yellow", "rectangle"
H.array = Tensor(Dim(2), Dim(2), 1/np.sqrt(2) * np.array([1.0, 1, 1, -1]))

CX = Z(1, 2) @ Id(1) >> Id(1) @ X(2, 1)
CZ = Z(1, 2) @ Id(1) >> Id(1) @ H @ Id(1) >> Id(1) @ Z(2, 1)


Swap = Id(2).swap(1, 1)

FSwap = Box('O', PRO(2), PRO(2), data=eval(Swap >> CZ).flatten(), draw_as_spider=True)
FSwap.array = FSwap.data
FSwap.color = 'white'
