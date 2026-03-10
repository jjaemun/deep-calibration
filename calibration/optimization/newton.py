# -*- coding: utf-8 -*-
# type: ignore


from ..functional import ContinuouslyDifferentiable


from .. import *


from .iterate import Iterate
from .stopping import Criterion


import torch


from typing import List


class NewtonRaphson(object):

    ''' Newton-Raphson root finding algorithm. '''

    def __init__(self, criterion: Criterion): 
       self.criterion = criterion


    def step(self, func: ContinuouslyDifferentiable, guess: ...) -> Iterate:

        ''' Computes single step. '''

        f = func(guess) 
        jacobian = func.jacobian(guess)
        x = (guess - f / jacobian)
        step = (guess - x)

        return Iterate(
            x = x, f = f, jacobian = jacobian, step = step
        )


    def optimize(self, func: ContinuouslyDifferentiable, guess: ...) -> List[Iterate]:

        ''' Computes full optimization. '''
        
        iterates: List[Iterate] = []
        while not self.criterion.check(iterates):
            iterates.append(self.step(func, guess if not iterates
                else iterates[-1].x))

        return iterates
