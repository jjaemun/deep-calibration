# -*- codi. '''
   
    @abc.abstractmethod
    def __call__(*args, **kwargs):
        raise NotImplementedError


class SchoutensCharfunc(charfunc):

    ''' Schoutens et. al. (2004), Albrecher et al. (2007)  modified Heston 
            characteristic function. '''
    
    def __call__(u: float, s: float, spot: float, rate: float, kappa: float, 
                 theta: float, sigma: float, rho: float, vol: float) -> complex:
        Q = kappa - sigma * rho * 1j * u
        D = numpy.sqrt(Q * Q + sigma * 
                sigma * u * (u + 1j))
        G = (Q - D) / (Q + D)

        # log-spot forward drift term.
        fdrift = 1j * u * (numpy.log(spot) + rate * s)

        # volatility mean-reversion drift term.
        vdrift = ((kappa * theta) / (sigma * sigma) * 
            ((Q - D) * s - 2 * numpy.log((1 - G * 
                numpy.exp(-D * s)) / (1 - G))))

        # variance sensitivity term.
        vsens = (vol / (sigma * sigma) * (Q - D) * 
            ((1 - numpy.exp(-D * s)) / (1 - G * 
                numpy.exp(-D * s))))                                          
        
        return numpy.exp(fdrift + vdrift + vsens)



class DelBañoRollinCharfunc(charfunc):

 ''' Corrected (see Cui et. al. (2016) [pp.7-8]) Del Baño Rollin et. al. (2010) 
            modified Heston characteristic function. '''

    def __call__(u: float, s: float, spot: float, rate: float, kappa: float, 
                 theta: float, sigma: float, rho: float, vol: float) -> complex:
        Q = kappa - sigma * rho * 1j * u
        D = numpy.sqrt(Q * Q + sigma * 
                sigma * u * (u + 1j))

        A = (u * (u + 1j) * numpy.sinh(D * s / 2) /
                (D * numpy.cosh(D * s / 2) + 
                    Q * numpy.sinh(D * s / 2)))

        B = (D * numpy.exp(kappa * s / 2) / 
                (D * numpy.cosh(D * s / 2) + 
                    Q * numpy.sinh(D * s / 2)))

        # log-spot forward drift term.
        fdrift = 1j * u * (numpy.log(spot) + rate * s)
        
        # volatility mean-reversion drift term.
        vdrift = (s * kappa * theta * rho * 
                    1j * u / sigma)
        
        # variance sensitivity term.
        vsens = vol * A

        factor = B ** (2 * kappa * theta / 
                    (sigma * sigma))

        return factor * numpy.exp(fdrift - vdrift - vsens)



class CuiCharfunc(charfunc):

    ''' Cui et. al. (2016) modified Heston characteristic function. '''

    def __call__(u: float, s: float, spot: float, rate: float, kappa: float, 
                 theta: float, sigma: float, rho: float, vol: float) -> complex:
        Q = kappa - sigma * rho * 1j * u
        D = numpy.sqrt(Q * Q + sigma * 
                sigma * u * (u + 1j))

        A = (u * (u + 1j) * numpy.sinh(D * s / 2) /
                numpy.exp((D * s / 2) + numpy.log(((D + Q) / 2) + 
                    numpy.exp(-D * s) * ((D - Q) / 2))))

        D = (numpy.log(D) + ((kappa - D) * s) / 2 - 
                numpy.log(((D + Q) / 2) + 
                    numpy.exp(-D * s) * ((D - Q) / 2)))

        # log-spot forward drift term.
        fdrift = 1j * u * (numpy.log(spot) + rate * s)
    
        # volatility mean-reversion drift term.
        vdrift = (s * kappa * theta * rho * 1j * 
                    u / sigma)

        # variance sensitivity term.
        vsens = - (vol * A) + (2 * kappa * theta /
                    (sigma * sigma)) * D

        return numpy.exp(fdrift - vdrift + vsens)

