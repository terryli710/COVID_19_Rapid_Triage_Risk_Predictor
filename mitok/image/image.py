import numpy
import cv2
import torch
import numbers
import torch.nn.functional as F
from mitok.utils.param_checker import is_image_tensor
from mitok.utils.param_checker import is_positive_real_number
from mitok.utils.param_checker import is_iterable
from mitok.utils.merror import MError


def image_tensor_resize(img_tensor, size=None, scale=None, mode='pytorch', return_type='numpy'):
    if mode == 'pytorch':
        #try:
        if True:
            assert isinstance(img_tensor, torch.Tensor) or isinstance(img_tensor, torch.cuda.FloatTensor) or isinstance(img_tensor, numpy.ndarray), \
                'image_tensor must be one of torch.Tensor, torch.cuda.FloatTensor, numpy.ndarray, get %s' % type(img_tensor)
            assert return_type in ['tensor', 'numpy', 'int16']
            if isinstance(img_tensor, numpy.ndarray):
                img_tensor = torch.Tensor(numpy.float32(img_tensor))
            img_shape = img_tensor.size()
            assert 5 >= len(img_shape) >= 3

            if size is None and scale is None:
                return MError(MError.E_FIELD_IMAGE, -1, 'one of size and scale should be set!'), None
            if scale is not None:
                if isinstance(scale, numbers.Number):
                    scale = (scale, scale, scale)
                assert isinstance(scale, tuple) or isinstance(scale, list)
                new_shape = [int(ori_size * scale) for (ori_size, scale) in zip(img_shape[-3:], scale)]
            elif size is not None:
                new_shape = size

            assert len(new_shape) == 3
            assert new_shape[0]*new_shape[1]*new_shape[2] > 0
            
            if not img_tensor.is_contiguous():
                img_tensor = img_tensor.contiguous()

            if new_shape[0] == img_shape[-3] and new_shape[1] == img_shape[-2] and new_shape[2] == img_shape[-1]:
                resize_tensor = img_tensor.view(new_shape[0], new_shape[1], new_shape[2])
                if return_type == 'numpy':
                    resize_tensor = resize_tensor.clamp(0, 255).cpu().numpy()
                    resize_tensor = numpy.uint8(resize_tensor)
                elif return_type == "int16":
                    resize_tensor = resize_tensor.cpu().numpy()
                    resize_tensor = numpy.int16(resize_tensor)
                return MError(MError.E_FIELD_IMAGE, 0, ''), resize_tensor
            if len(img_shape) < 5:
                img_d, img_h, img_w = img_shape[-3:]
                img_tensor = img_tensor.view(1, 1, img_d, img_h, img_w)
            resize_tensor = F.interpolate(img_tensor, new_shape, mode='trilinear', align_corners=True).data[0, 0]

            if return_type == 'numpy':
                resize_tensor = resize_tensor.clamp(0, 255).cpu().numpy()
                #resize_tensor = numpy.rint(resize_tensor, out=resize_tensor)
                #resize_tensor = numpy.clip(resize_tensor, 0, 255, out=resize_tensor)
                resize_tensor[resize_tensor > 0] = 1
                resize_tensor = numpy.uint8(resize_tensor)
            elif return_type == "int16":
                resize_tensor = resize_tensor.cpu().numpy()
                resize_tensor = numpy.rint(resize_tensor, out=resize_tensor)
                resize_tensor = numpy.int16(resize_tensor)
            torch.cuda.empty_cache()

            return MError(MError.E_FIELD_IMAGE, 0, ''), resize_tensor
        #except Exception as ex:
        #    return MError(MError.E_FIELD_IMAGE, -1, ex.message), None
    else:
        # check img_tensor
        msg, flag = is_image_tensor(img_tensor)
        if not flag:
            return MError(MError.E_FIELD_IMAGE, -1, msg), None
        d, h, w = img_tensor.shape
        new_d, new_h, new_w = d, h, w
        # check size
        if size is None and scale is None:
            return MError(MError.E_FIELD_IMAGE, -1, 'one of size and scale should be set!'), None
        if not size is None:
            if not is_iterable(size):
                return MError(MError.E_FIELD_IMAGE, -1, 'param type error: size is %s, expected to be iterable'
                              % type(size)), None
            if len(size) != 3:
                return MError(MError.E_FIELD_IMAGE, -1, 'size has %d elements, expected to be 3' % len(size)), None
            if not (is_positive_real_number(size[0]) and is_positive_real_number(size[1]) and is_positive_real_number(size[2])):
                return MError(MError.E_FIELD_IMAGE, -1, 'elements of size are %s, expected to be positive real numbers'
                              % str((type(x) for x in size))), None
            new_d, new_h, new_w = size
        elif not scale is None:
            if is_positive_real_number(scale):
                scale = (scale, scale, scale)
            elif is_iterable(scale):
                if len(scale) != 3:
                    return MError(MError.E_FIELD_IMAGE, -1, 'scale has %d elements, expected to be 3' % len(scale)), None
                if not (is_positive_real_number(scale[0]) and is_positive_real_number(scale[1]) and is_positive_real_number(scale[2])):
                    return MError(MError.E_FIELD_IMAGE, -1, 'elements of scale are %s, expected to be positive real numbers'
                                  % str((type(x) for x in scale))), None
            else:
                return MError(MError.E_FIELD_IMAGE, -1, 'param type error: scale is %s, expected to be positive real number or tuple'
                              % type(scale)), None
            new_d, new_h, new_w = int(d * scale[0]), int(h * scale[1]), int(w * scale[2])
        new_shape = (new_d, new_h, new_w)
        if d == new_d and h == new_h and w == new_w:
            return MError(MError.E_FIELD_IMAGE, 0, ''), img_tensor
        tmp_img_tensor = numpy.zeros([img_tensor.shape[0], new_shape[1], new_shape[2]], dtype=img_tensor.dtype)
        new_img_tensor = numpy.zeros(new_shape, dtype=img_tensor.dtype)
        for idx in range(img_tensor.shape[0]):
            tmp_img_tensor[idx, :, :] = cv2.resize(img_tensor[idx, :, :], (new_shape[2], new_shape[1]), interpolation=cv2.INTER_CUBIC)
        for idx in range(new_shape[1]):
            new_img_tensor[:, idx, :] = cv2.resize(tmp_img_tensor[:, idx, :], (new_shape[2], new_shape[0]), interpolation=cv2.INTER_CUBIC)
        return MError(MError.E_FIELD_IMAGE, 0, ''), new_img_tensor
