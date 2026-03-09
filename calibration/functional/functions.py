# -*- coding: utf-8 -*-


import abc


class function(abc.ABC):

    ''' Function abstraction. '''

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class ContinuouslyDifferentiable(function):

    ''' Continously differentiable function. '''

    @abc.abstractmethod
    def jacobian(self, *args, **kwargs):
        raise NotImplementedError


class TwiceContinuouslyDifferentiable(ContinuouslyDifferentiable):

    ''' Twice continously differentiable function. '''

    @abc.abstractmethod
    def hessian(self, *args, **kwargs):
        raise NotImplementedError

