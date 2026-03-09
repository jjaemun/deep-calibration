# -*- coding: utf-8 -*- 
# type: ignore


from ..constants import *

import abc
import torch


class charfunc(abc.ABC):
    
    ''' Characteristic function. '''

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class SchoutensCharfunc(charfunc):

    ''' Schoutens et. al. (2004), Albrecher et al. (2007)  modified Heston 
            characteristic function. '''
    
    def __call__(self, spot: f64, rate: f64, kappa: f64, theta: f64, 
                 sigma: f64, rho: f64, vol: f64, u: f64, s: f64) -> c128:
        Q = kappa - sigma * rho * J * u
        D = torch.sqrt(Q * Q + sigma * 
                sigma * u * (u + J))
        G = (Q - D) / (Q + D)

        # log-spot forward drift term.
        fdrift = J * u * (torch.log(spot) + rate * s)

        # volatility mean-reversion drift term.
        vdrift = ((kappa * theta) / (sigma * sigma) * 
            ((Q - D) * s - 2 * torch.log((1 - G * 
                torch.exp(-D * s)) / (1 - G))))

        # variance sensitivity term.
        vsens = (vol / (sigma * sigma) * (Q - D) * 
            ((1 - torch.exp(-D * s)) / (1 - G * 
                torch.exp(-D * s))))                                          
        
        return torch.exp(fdrift + vdrift + vsens)



class DelBañoRollinCharfunc(charfunc):

    ''' Corrected (see Cui et. al. (2016) [pp.7-8]) Del Baño Rollin et. al. (2010) 
            modified Heston characteristic function. '''

    def __call__(self, spot: f64, rate: f64, kappa: f64, theta: f64, 
                 sigma: f64, rho: f64, vol: f64, u: f64, s: f64) -> c128:
        Q = kappa - sigma * rho * J * u
        D = torch.sqrt(Q * Q + sigma * 
                sigma * u * (u + J))

        A = (u * (u + J) * torch.sinh(D * s / 2) /
                (D * torch.cosh(D * s / 2) + 
                    Q * torch.sinh(D * s / 2)))

        B = (D * torch.exp(kappa * s / 2) / 
                (D * torch.cosh(D * s / 2) + 
                    Q * torch.sinh(D * s / 2)))

        # log-spot forward drift term.
        fdrift = J * u * (torch.log(spot) + rate * s)
        
        # volatility mean-reversion drift term.
        vdrift = (s * kappa * theta * rho * 
                    J * u / sigma)
        
        # variance sensitivity term.
        vsens = vol * A

        factor = B ** (2 * kappa * theta / 
                    (sigma * sigma))

        return factor * torch.exp(fdrift - vdrift - vsens)



class CuiCharfunc(charfunc):

    ''' Cui et. al. (2016) modified Heston characteristic function. '''

    def __call__(self, spot: f64, rate: f64, kappa: f64, theta: f64, 
                 sigma: f64, rho: f64, vol: f64, u: f64, s: f64) -> c128:
        Q = kappa - sigma * rho * J * u
        D = torch.sqrt(Q * Q + sigma * 
                sigma * u * (u + J))

        A = (u * (u + J) * torch.sinh(D * s / 2) /
                torch.exp((D * s / 2) + torch.log(((D + Q) / 2) + 
                    torch.exp(-D * s) * ((D - Q) / 2))))

        # in the paper this is D; running out of letters here...
        B = (torch.log(D) + ((kappa - D) * s) / 2 - 
                torch.log(((D + Q) / 2) + 
                    torch.exp(-D * s) * ((D - Q) / 2)))

        # log-spot forward drift term.
        fdrift = J * u * (torch.log(spot) + rate * s)
    
        # volatility mean-reversion drift term.
        vdrift = (s * kappa * theta * rho * J * 
                    u / sigma)

        # variance sensitivity term.
        vsens = - (vol * A) + (2 * kappa * theta /
                    (sigma * sigma)) * B

        return torch.exp(fdrift - vdrift + vsens)

