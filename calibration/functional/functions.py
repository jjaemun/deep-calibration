# -*- coding: utf-8 -*-


import abc


class function(abc.ABC):

    ''' Function abstraction. '''

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class Differentiable(function):

    ''' Continously differentiable function. '''

    @abc.abstractmethod
    def jacobian(self, *args, **kwargs):
        raise NotImplementedError


class TwiceDifferentiable(Differentiable):

    ''' Twice continously differentiable function. '''

    @abc.abstractmethod
    def hessian(self, *args, **kwargs):
        raise NotImplementedError

