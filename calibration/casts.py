# -*- coding: utf-8 -*-

import torch
import typing


def cast(*tensors: ..., to: typing.Type) -> typing.Tuple[...]:

    ''' Casts all input tensors to a different type. '''
    
    return tuple(
        tensor.to(to) for tensor in tensors
    )
