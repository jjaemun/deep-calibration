# -*- coding: utf-8 -*-
# type: ignore


from .. import *


import abc
import torch


from .iterate import Iterate


from typing import List, Optional


class Criterion(abc.ABC):
    
    ''' Criterion abstraction. '''

    @abc.abstractmethod
    def check(self, iterates: List[Iterate]) -> bool:
        raise NotImplementedError



class StoppingCriterion(Criterion):

    ''' Composite stopping criterion abstraction. '''

    def __init__(self, criterions: List[Criterion]):
        self.criterions = criterions

    def check(self, iterates: List[Iterate]) -> bool:
        
        ''' Checks criterion violations. '''

        kill: bool = False
        for criterion in self.criterions:
            kill = kill or criterion.check(iterates)

        return kill


class MaxIterations(Criterion):

    ''' Limits the number of iterations optimizer 
            can take. '''

    def __init__(self, iterations: Optional[u32] = 100):
        self.iterations = iterations

    def check(self, iterates: List[Iterate]):
        return len(iterates) >= self.iterations


class EvaluationNorm(Criterion):

    ''' Stops when function evaluation norm fallss below
            tolerance. '''

    def __init__(self, eps: Optional[f64] = 1e-6):
        self.eps = eps

    def check(self, iterates: List[Iterate]):
        return torch.norm(iterates[-1].f) < self.eps


class StepNorm(Criterion):

    ''' Stops when step norm falls below 
            tolerance. '''

    def __init__(self, eps: Optional[f64] = 1e-6):
        self.eps = eps

    def check(self, iterates: List[Iterate]):
        return torch.norm(iterates[-1].step) < self.eps


class JacobianNorm(Criterion):

    ''' Stops when jacobian norm falls below 
            tolerance. '''

    def __init__(self, eps: Optional[f64] = 1e-6):
        self.eps = eps

    def check(self, iterates: List[Iterate]):
        return torch.norm(iterates[-1].jacobian) < self.eps



class RelativeImprovementNorm(Criterion):

    ''' Stops when relative improvement norm falls below 
            tolerance. '''

    def __init__(self, eps: Optional[f64] = 1e-6):
        self.eps = eps

    def check(self, iterates: List[Iterate]):
        # first iteration!
        if len(iterates <= 1):
            return False

        prev = torch.norm(iterates[-2].f)
        curr = torch.norm(iterates[-1].f)

        improvement = (prev - curr) / prev

        return improvement < self.eps
