# -*- coding: utf-8 -*-
# type: ignore 

from ..options import Option


from .. import *


import torch



class BlackScholes(object):

    ''' Black-Scholes model. '''
    
    gaussian = torch.distributions.Normal(0.0, 1.0)

    def __init__(self, spot: f64, rate: f64, sigma: f64):
        self.spot = spot
        self.rate = rate
        self.sigma = sigma 

    def price(self, option: Option) -> f64:

        ''' Computes option price. '''

        forward = (torch.exp(self.rate * option.tenor)
                      * self.spot)
        
        d1 = ((torch.log(forward / option.strike) + 
                (self.sigma * self.sigma / 2) * option.tenor) / 
                    (self.sigma * torch.sqrt(option.tenor)))

        d2 = d1 - self.sigma * torch.sqrt(option.tenor)

        return (torch.exp(-self.rate * option.tenor) *  (forward * 
            self.gaussian.cdf(d1) - option.strike * self.gaussian.cdf(d2)))

