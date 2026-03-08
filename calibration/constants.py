# -*- coding: utf-8 -*-
# type: ignore


from .hints import *

import torch


'''  constants!  '''
ZERO: Tensor[f64]  = torch.tensor(0.0, dtype = f64)
EPS:  Tensor[f64]  = torch.tensor(1e-8, dtype = f64)
ONE:  Tensor[f64]  = torch.tensor(1.0, dtype = f64)
J:    Tensor[c128] = torch.tensor(1j, dtype = c128)
PI:   Tensor[f64]  = torch.acos(torch.tensor(-1.0, dtype = f64))
UB:   Tensor[f64]  = torch.tensor(1e3, dtype = f64)


