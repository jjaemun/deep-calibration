# -*- coding: utf-8 -*-

from ..quadrature import QuadratureScheme
from ..options import Option


from .. import *


from functools import partial
import torch


from typing import Callable



class Heston(object):

    ''' Heston local stochastic volatility model . '''

    def __init__(self, spot: f64, rate: f64, kappa: f64, theta: f64, 
                       sigma: f64, rho: f64, vol: f64, cf: Callable):
        self.spot = spot
        self.rate = rate
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma 
        self.rho = rho
        self.vol = vol

        self.cf = partial(cf, spot = spot, rate = rate, kappa = kappa,
                    theta = theta, sigma = sigma, rho = rho, vol = vol)


    def price(self, option: Option, scheme: QuadratureScheme) -> f64:

        ''' Computes option price. '''

        discount = torch.exp(-self.rate * option.tenor)
        intrinsic = (self.spot - discount * option.strike)

        # closure!
        def integrand(u: Tensor[f64]):
            common = (torch.exp(-u * J * torch.log(option.strike)) / 
                (J * u))
            return ((common *  self.cf(u - J, option.tenor)).real
                - option.strike * (common * self.cf(u, option.tenor)).real)
              
        return (intrinsic / 2 + (torch.div(discount, PI)) * 
            scheme.integrate(integrand))
