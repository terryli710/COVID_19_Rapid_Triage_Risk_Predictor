# -*- coding: utf-8 -*-
import  os, sys
import  gdcm
import numpy
import traceback

def get_gdcm_to_numpy_typemap():
    """Returns the GDCM Pixel Format to numpy array type mapping."""
    _gdcm_np = {gdcm.PixelFormat.UINT8: numpy.uint8,
                gdcm.PixelFormat.INT8: numpy.int8,
                # gdcm.PixelFormat.UINT12 :numpy.uint12,
                # gdcm.PixelFormat.INT12    :numpy.int12,
                gdcm.PixelFormat.UINT16: numpy.uint16,
                gdcm.PixelFormat.INT16: numpy.int16,
                gdcm.PixelFormat.UINT32: numpy.uint32,
                gdcm.PixelFormat.INT32: numpy.int32,
                # gdcm.PixelFormat.FLOAT16:numpy.float16,
                gdcm.PixelFormat.FLOAT32: numpy.float32,
                gdcm.PixelFormat.FLOAT64: numpy.float64}
    return _gdcm_np


def get_numpy_array_type(gdcm_pixel_format):
    """Returns a numpy array typecode given a GDCM Pixel Format."""
    return get_gdcm_to_numpy_typemap()[gdcm_pixel_format]


def gdcm_to_numpy(image):
    pf = image.GetPixelFormat()

    assert pf.GetScalarType() in get_gdcm_to_numpy_typemap().keys(), \
        "Unsupported array type %s" % pf
    dtype = get_numpy_array_type(pf.GetScalarType())
    gdcm_array = image.GetBuffer()
    if sys.version_info[0] < 3:
        result = numpy.frombuffer(gdcm_array, dtype=dtype)
    else:
       result = numpy.frombuffer(gdcm_array.encode("utf-8", "surrogateescape"), dtype=dtype)
    # result.shape = shape
    dim = image.GetDimension(1), image.GetDimension(0)

    assert result.size == (dim[0] * dim[1]), "size not match"

    result.shape = dim
    return result

class pygdcm(object):
    gdcm_Reader = None
    def __init__(self, dcm_path):
        try:
            self.dcm_path = dcm_path
            pygdcm.gdcm_Reader = gdcm.ImageReader()
            pygdcm.gdcm_Reader.SetFileName(self.dcm_path)
            file = pygdcm.gdcm_Reader.GetFile()
            self.dataset = file.GetDataSet()
            self.strFilter = gdcm.StringFilter()
            self.strFilter.SetFile(pygdcm.gdcm_Reader.GetFile())
            self.image_u16 = []
        except Exception as e:
            print(traceback.print_exc())
            return

    def loadCheckDcmFile(self):
        try:
            if not pygdcm.gdcm_Reader.Read():
                print('readerr')
                return False
            self.image_u16 = (gdcm_to_numpy(pygdcm.gdcm_Reader.GetImage()))
            return True
        except Exception as e:
            print(traceback.print_exc())
            print('load excpetion')
            return False
    
    def getTagVM(self, Tag_a, Tag_b):
        try:
            if not self.dataset.FindDataElement(gdcm.Tag(Tag_a,Tag_b)):
                print('getTagVM Error:No Tag(%x,%x)'%(Tag_a,Tag_b))
                return -1
            value = self.dataset.GetDataElement(gdcm.Tag(Tag_a, Tag_b))
            return gdcm.Global().GetDicts().GetDictEntry(value.GetTag()).GetVM().GetLength()
        except Exception as e:
            print(traceback.print_exc())
            return -1

    def getTagData(self, Tag_a, Tag_b):
        try:
            if not self.dataset.FindDataElement(gdcm.Tag(Tag_a,Tag_b)):
                return None
            return self.strFilter.ToStringPair(gdcm.Tag(Tag_a,Tag_b))[1]
        except Exception as e:
            print(traceback.print_exc())
            return None
            
    def getImageArray(self):
        return self.image_u16
        try:
            image_u16.append(self.image_u16)
            if image_u16 == []:
                return False
            else:
                return True
        except Exception as e:
            traceback.print_exc()
            return False

#main
def DcmHandlerDW(dcm_path):
    try:
        dcm = pygdcm(dcm_path)
        if dcm is not None:
            return dcm
        else:
            return None
    except Exception as e:
        print(traceback.print_exc())
        return None
