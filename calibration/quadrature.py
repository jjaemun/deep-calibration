# -*- coding: utf-8 -*-
# type: ignore


from . import *

import abc
import numpy
import torch

from typing import Callable


class QuadratureScheme(abc.ABC):

    ''' Numerical integration scheme. '''
    
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


    @abc.abstractmethod
    def integrate(self, f: Callable, a: f64, b: f64):
        raise NotImplementedError



class GaussLegendreQuadrature(QuadratureScheme):
    
    ''' Gauss-Legendre quadrature suitable for semi-infinite 
            domain.'''

    def __init__(self, nodes: u32, a: f64, b: f64):
        assert (nodes >= 1), 'recommended 64 nodes.'

        abscissa, weights = numpy.polynomial.legendre.leggauss(nodes)
       
        self.abscissa = cast(torch.from_numpy(abscissa), to = f64)
        self.weights = cast(torch.from_numpy(weights), to = f64)


    def integrate(self, func: Callable, a: Optional[f64] = ZERO, b: Optional[f64] = UB) -> f64:

        ''' Computes Gauss-Legendre quadrature for func. '''

        width: f64 = (b - a)
        support: f64 = (a + b)
   
        affine: torch.Tensor = (support / 2 + 
            width / 2 * self.abscissa)
        observations: torch.Tensor = func(affine)

        return (torch.dot(observations, self.weights)
            * (width / 2))

