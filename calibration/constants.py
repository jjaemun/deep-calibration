# -*- coding: utf-8 -*-
# type: ignore


from .hints import *

import torch


'''  constants!  '''
ZERO: f64 = torch.tensor(0.0, dtype = f64)
ONE: f64 = torch.tensor(1.0, dtype = f64)
PI: f64 = torch.acos(torch.tensor(-1.0, dtype = f64))
UB: f64 = torch.tensor(1e3, dtype = f64)
EPS: f64 = torch.tensor(1e-8, dtype = f64)
J: c128 = torch.tensor(1j, dtype = c128)


