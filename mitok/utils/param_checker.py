from collections import Iterable
import numpy
import sys

def is_string(s):
    if int(sys.version[0]) < 3:
        return type(s) in [str, unicode]
    else:
        return type(s) in [str]


def is_iterable(a):
    return isinstance(a, Iterable)


def is_numpy_array(a):
    #return type(a) is numpy.ndarray
    return isinstance(a, numpy.ndarray)


def is_real_number(a):
    return type(a) in [int, float]


def is_positive_real_number(a):
    return is_real_number(a) and a > 0


def is_positive_int_number(a):
    return (type(a) in [int]) and (a > 0)


def is_image_tensor(image_tensor):
    # image_tensor should be a 3d uint8 numpy.ndarray
    if not is_numpy_array(image_tensor):
        return 'param type error: image_tensor is %s, expected to be numpy.ndarray' % type(image_tensor).__name__, False
    if image_tensor.dtype != numpy.uint8:
        return 'numpy dtype error: image_tensor.dtype is %s, expected to be uint8' % str(image_tensor.dtype), False
    if len(image_tensor.shape) != 3:
        return 'shape of image_tensor is %s, expected to be 3d-tensor' % str(image_tensor.shape), False
    return '', True
