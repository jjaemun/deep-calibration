# -*- coding: utf-8 -*-
# type: ignore

import torch


'''  integral!  '''
u16 = torch.uint16
u32 = torch.uint32
u64 = torch.uint64

i16 = torch.int16
i32 = torch.int32
i64 = torch.int64


'''  fp!  '''
f32 = torch.float32
f64 = torch.float64


'''  complex!  '''
c64 = torch.complex64
c128 = torch.complex128


'''  constants!  '''
EPS: f64 = torch.tensor(1e-8, dtype = f64)
PI: f64 = torch.acos(torch.tensor(-1.0, dtype = f64))
J: c128 = torch.tensor(1j, dtype = c128)

