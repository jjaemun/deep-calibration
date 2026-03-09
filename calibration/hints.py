# -*- coding: utf-8 -*-
# type: ignore

import torch
import typing


# types!
u16 = torch.uint16
u32 = torch.uint32
u64 = torch.uint64

i16 = torch.int16
i32 = torch.int32
i64 = torch.int64

f32 = torch.float32
f64 = torch.float64

c64 = torch.complex64
c128 = torch.complex128


# tensor!
class Tensor(object):

    ''' Type hinting helper. '''
    
    def __class_getitem__(cls, dtype: torch.dtype) -> typing.Type:
        
        ''' Type hinting clearly does not enforce behavior but
                helps reason about code. '''

        return typing.Annotated[torch.Tensor, dtype]
