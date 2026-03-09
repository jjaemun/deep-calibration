# -*- coding: utf-8 -*-
#type: ignore 


from .. import *


import torch


from typing import (
    Mapping, 
    Optional, 
    Tuple
)


class Sampler(object):

    ''' General wrapper for sampling engines to generate synthetic 
            calibration data. '''

    def __init__(self, dim: i32, bounds: Mapping[str, Tuple[f64, f64]], 
                 engine: ..., seed: Optional[int] = None):
        assert len(list(bounds.keys())) == dim

        self.dim = dim
        self.bounds = bounds

        self.engine = engine(
            dimension = dim, scramble = True, seed = seed
        )


    def random(self, n: u32) -> Tensor[f64]:

        ''' Draws n samples from specified input space '''

        sequences = self.engine.draw(n = n, dtype = f64)
        for i, (a, b) in enumerate(list(self.bounds.values())):
            sequences[..., i] = a + ((b - a) * 
                sequences[..., i])

        return sequences

   
    def sample(self, n: u32, feller: bool = False) -> Tensor[f64]:

        ''' Builds scaled samples optionally satisfying 
                feller cond. '''

        return self.random(n) if not feller else self.feller(self.random(n))


    def feller(self, sequences: Tensor[f64]) -> Tensor[f64]:

        ''' Filters feller cond. Generated samples can be less than n! 
                In practice it is often violated, so this is insignificant.  '''

        keys = list(self.bounds.keys())
        indices = {
            key : idx 
                for idx, key in enumerate(keys)
        }

        fields = set(('kappa', 'theta', 'sigma'))
        if not fields.issubset(keys):
            raise ValueError('fields do not match feller condition.')

        kappa = sequences[..., indices['kappa']]
        theta = sequences[..., indices['theta']]
        sigma = sequences[..., indices['sigma']]

        mask: Tensor[bool] = 2 * kappa * theta > sigma * sigma

        return sequences[mask]
