# -*- coding: utf-8 -*-
# type: ignore


from .hints import Tensor, f64

import abc
import torch


class Option(abc.ABC):

    ''' Contingent claim abstraction. '''

    @abc.abstractmethod
    def payoff(self, *args, **kwargs):
        raise NotImplementedError


class EuropeanCall(Option):

    ''' Contingent claim abstraction.'''
    
    def __init__(self, strike: f64, tenor: f64):
        self.strike = strike 
        self.tenor = tenor

    def payoff(self, s: Tensor[f64]) -> Tensor[f64]:
        return torch.clamp((s - self.strike), min= 0.0)


class EuropeanPut(Option):

    ''' Contingent claim abstraction.'''
    
    def __init__(self, strike: f64, tenor: f64):
        self.strike = strike 
        self.tenor = tenor

    def payoff(self, s: Tensor[f64]) -> Tensor[f64]:
        return torch.clamp((self.strike - s), min= 0.0)
