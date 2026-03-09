# -*- coding: utf-8 -*-
#type: ignore 


from .. import *

from dataclasses import dataclass


@dataclass
class Iterate(object):
    
    ''' Optimization iterate. '''

    x: Tensor[f64]
    f: Tensor[f64] 
    jacobian: Tensor[f64]
    step: Tensor[f64]
