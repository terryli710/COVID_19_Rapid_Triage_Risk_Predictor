# -*- coding: utf-8 -*-
import os
import glob
import traceback
from scipy import stats
import numpy

from mitok.image.image import image_tensor_resize
from mitok.utils.merror import MError
import copy
################### 3rd_party dicom #####################
from . import pygdcm


def parse_dicom_window(dcm):
    if not dcm:
        raise(Exception("error in get window"))
    wc, ww = None, None

    wc_vm = dcm.getTagVM(0x0028, 0x1050)
    ww_vm = dcm.getTagVM(0x0028, 0x1051)
    if wc_vm >= 0:
        val = dcm.getTagData(0x0028, 0x1050)
        wc = float(val.split('\\')[0])
    else:
        raise(Exception("error in get window"))
    if ww_vm >= 0:
        val = dcm.getTagData(0x0028, 0x1051)
        ww = float(val.split('\\')[0])
    else:
        raise(Exception("error in get window"))
    return wc, ww


class DICOM(object):
    '''
    initialize dcm data 
    :dcm_dir：the path of file
    :dcm_file：the name of file
    :return:Null
    '''

    def __init__(self, dcm_path, b_ct = True):
        self.err_list = []
        self.warning_list = []
        self.is_valid_ = True
        self.pid = None
        self.study_uid = None
        self.image_loc = None
        self.ctdivol = None
        self.body_part = None
        self.manufacturer = None
        self.kernel = None
        self.kvp = None
        self.pat_pos = None
        self.b_ct = b_ct

        self.img_height = None
        self.img_width = None

        self.series_uid = None

        self.window_center = None
        self.window_width = None
        self.image_position = None
        self.patient_position = None

        if type(dcm_path) != str:
            self.is_valid_ = False
            raise(Exception("[Error] DICOM File Path type error"))

        self.dcm_path = dcm_path
        self.dcm = pygdcm.DcmHandlerDW(self.dcm_path)
        ret = self.dcm.loadCheckDcmFile()
        if not ret:
            self.is_valid_ = False
            self.err_list.append(MError(MError.E_FIELD_DICOM, 'E_DICOM_002','[Error] DICOM load error，dcm_file:%s.\n'%(self.dcm_path)))
            return

        try:
            val = self.dcm.getTagData(0x0010, 0x0020)
            if val is None or val == '':
                self.pid = ""
            else:
                self.pid = val

            val = self.dcm.getTagData(0x0020, 0x000d)
            if val is None or val == '':
                self.study_uid = ''
                self.is_valid_ = False
                self.err_list.append(MError(MError.E_FIELD_DICOM, 'E_DICOM_003', '[Error] DICOM Study Instance UID erro，dcm_path:%s.\n' % self.dcm_path))
                return
            else:
                self.study_uid = val

            val = self.dcm.getTagData(0x0020, 0x000e)
            if val is None or val == '':
                self.series_uid = None
                self.is_valid_ = False
                self.err_list.append(MError(MError.E_FIELD_DICOM, 'E_DICOM_004', '[Error] DICOM Series Instance UID error，dcm_path:%s.\n' % self.dcm_path))
                return
            else:
                self.series_uid = val

            val = self.dcm.getTagData(0x0020, 0x0013)
            if val is None or val == '':
                self.instance_no = None
                self.is_valid_ = False
                self.err_list.append(MError(MError.E_FIELD_DICOM, 'E_DICOM_005', '[Error] DICOM Instance Number error，dcm_path:%s.\n' % self.dcm_path))
                return
            else:
                self.instance_no = int(val)
            val = self.dcm.getTagData(0x0028, 0x0010)
            if val is None or val == '':
                if self.dcm.image_u16 == []:
                    self.img_height = None
                    self.is_valid_ = False
                    self.err_list.append(MError(MError.E_FIELD_DICOM, 'E_DICOM_006', '[Error] DICOM error in get image array with img_height，dcm_path:%s.\n' % self.dcm_path))
                    return
                self.img_height = int(numpy.array(self.dcm.image_u16).shape[-1])
            else:
                self.img_height = int(val)
            val = self.dcm.getTagData(0x0028, 0x0011)
            if val is None or val == '':
                if self.dcm.image_u16 == []:
                    self.img_width = None
                    self.is_valid_ = False
                    self.err_list.append(MError(MError.E_FIELD_DICOM, 'E_DICOM_007', '[Error] DICOM error in get image array with img_width，dcm_path:%s.\n' % self.dcm_path))
                    return
                self.img_width = int(numpy.array(self.dcm.image_u16).shape[-2])
            else:
                self.img_width = int(val)
            val = self.dcm.getTagData(0x0028, 0x0030)
            vm = self.dcm.getTagVM(0x0028, 0x0030)
            if val is None or val == '' or vm != 2:
                val = self.dcm.getTagData(0x0018, 0x1164)
                vm = self.dcm.getTagVM(0x0018,0x1164)
                if val is None or val == '' or vm != 2:
                    if b_ct:
                        self.is_valid_ = False
                        self.err_list.append(MError(MError.E_FIELD_DICOM, 'E_DICOM_008', '[Error] DICOM error in get pixel spacing，dcm_path:%s, val is None: %s, vm != 2:%s.\n' %(
                            self.dcm_path, val is None, vm != 2)))
                        return
                    else:
                        self.pixel_spacing = None
                else:
                    self.pixel_spacing = [float(v) for v in val.split('\\')]
            else:
                self.pixel_spacing = [float(v) for v in val.split('\\')]

            val = self.dcm.getTagData(0x0018, 0x0050)
            if val is None or val == '':
                self.warning_list.append(MError(MError.E_FIELD_DICOM, 'W_DICOM_001', '[Warning] DICOM Warning in get slice thickness，dcm_path:%s.\n' % self.dcm_path))
                self.slice_thickness = -1
            else:
                self.slice_thickness = float(val)

            val = self.dcm.getTagData(0x0020, 0x0032)
            if val is None or val == '' or self.dcm.getTagVM(0x0020, 0x0032) != 3:
                self.image_loc = None
                if b_ct:
                    self.err_list.append(MError(MError.E_FIELD_DICOM, 'E_DICOM_014', '[Error] DICOM Error in get image_loc，dcm_path:%s.\n' % self.dcm_path))
                    self.is_valid_ = False
            else:
                self.image_loc =[float(v) for v in val.split('\\')]
            try:
                val = self.dcm.getTagData(0x0020, 0x1041)
                if val is None or val == '':
                    if not (self.image_loc is None):
                        if len(self.image_loc) > 2:
                            self.slice_loc = self.image_loc[2]
                        else:
                            self.slice_loc = None
                            if b_ct:
                                self.is_valid_ = False
                                self.err_list.append(MError(MError.E_FIELD_DICOM, 'E_DICOM_010',
                                                        '[Error] DICOM error in get slice_loc，dcm_path:%s.\n' % self.dcm_path))
                    else:
                        self.slice_loc = None
                        if b_ct:
                            self.is_valid_ = False
                            self.err_list.append(MError(MError.E_FIELD_DICOM, 'E_DICOM_009', '[Error] DICOM error in get image position，dcm_path:%s.\n' % self.dcm_path))
                else:
                    self.slice_loc = float(val)
            except Exception as e:
                print(traceback.format_exc())
                self.slice_loc = None
                if b_ct:
                    self.is_valid_ = False
                    self.err_list.append(MError(MError.E_FIELD_DICOM, 'E_DICOM_010', '[Error] DICOM error in get slice_loc，dcm_path:%s.\n' % self.dcm_path))
            ########

            val = self.dcm.getTagData(0x0018, 0x9345)
            if val is None or val == '':
                self.ctdivol = None
                self.warning_list.append(MError(MError.E_FIELD_DICOM, 'W_DICOM_003',
                                           '[Warning] DICOM Warning has no CTDIvol，dcm_path:%s.\n' % self.dcm_path))
            else:
                self.ctdivol = (val)

            val = self.dcm.getTagData(0x0018, 0x0015)
            if val is None or val == '':
                self.body_part = None
                self.warning_list.append(MError(MError.E_FIELD_DICOM, 'W_DICOM_004',
                                           '[Warning] DICOM Warning has no Body Part，dcm_path:%s.\n' % self.dcm_path))
            else:
                self.body_part = (val)

            val = self.dcm.getTagData(0x0008, 0x0070)
            if val is None or val == '':
                self.manufacturer = None
                self.warning_list.append(MError(MError.E_FIELD_DICOM, 'W_DICOM_005',
                                           '[Warning] DICOM Warning has no Manufacturer，dcm_path:%s.\n' % self.dcm_path))
            else:
                self.manufacturer = (val)

            val = self.dcm.getTagData(0x0018, 0x1210)
            if val is None or val == '':
                self.kernel = None
                self.warning_list.append(MError(MError.E_FIELD_DICOM, 'W_DICOM_006',
                                           '[Warning] DICOM Warning has no Convolutional Kernel，dcm_path:%s.\n' % self.dcm_path))
            else:
                self.kernel = (val)

            val = self.dcm.getTagData(0x0018, 0x0060)
            if val is None or val == '':
                self.kvp = None
                self.warning_list.append(MError(MError.E_FIELD_DICOM, 'W_DICOM_007',
                                           '[Warning] DICOM Warning has no Kvp，dcm_path:%s.\n' % self.dcm_path))
            else:
                self.kvp = (val)

            val = self.dcm.getTagData(0x0028, 0x1053)
            if val is None or val == '':
                self.slope = 1.0
            else:
                self.slope = float(val)

            val = self.dcm.getTagData(0x0028, 0x1052)
            if val is None or val == '':
                self.intercept = 0.0
            else:
                self.intercept = float(val)

            val = self.dcm.getTagData(0x2020,0x0010)
            if val is None or val == '':
                self.image_position = None
            else:
                self.image_position = str(val)

            val = self.dcm.getTagData(0x0018,0x5100)
            if val is None or val == '':
                self.patient_position = None
            else:
                self.patient_position = str(val)

            try:
                self.window_center, self.window_width = parse_dicom_window(self.dcm)
            except Exception as ex:
                if b_ct:
                    print(traceback.print_exc())
                    self.is_valid_ = False
                    self.err_list.append(MError(MError.E_FIELD_DICOM, 'E_DICOM_015', '[Error] error in get window_center and window_width'))


            val = self.dcm.getTagData(0x0028,0x0103)
            if val is not None:
                self.pixelRepresentation = int(val)
            else:
                self.pixelRepresentation = 0

            val = self.dcm.getTagData(0x0028,0x0101)
            if val is not None:
                self.bitsStored = int(val)
            else:
                self.bitsStored = None

        except Exception as ex:
            print(traceback.print_exc())
            self.is_valid_ = False
            self.err_list.append(MError(MError.E_FIELD_DICOM, 'E_DICOM_013', '[Exception DICOM] dcm_path:%s ,error_info:%s.\n' %(self.dcm_path,str(ex))))

    def is_valid(self):
        return self.is_valid_

    def get_image(self):
        """
        Extract image from dicom
        """
        if not self.is_valid():
            return MError(MError.E_FIELD_DICOM, 'E_DICOM_011','[Error] DICOM is not valid, dcm_path:%s.\n' %self.dcm_path), None

        if self.dcm.image_u16 == []:
            return MError(MError.E_FIELD_DICOM, 'E_DICOM_012', '[Error] Get image array failed, dcm_path:%s.\n' %self.dcm_path), None
        img_int16 = numpy.array(self.dcm.image_u16, dtype=self.dcm.image_u16.dtype) * self.slope + self.intercept

        if self.pixelRepresentation == 1 and self.bitsStored is not None:
            shift = 32 - self.bitsStored
            maxint = 2147483647

            img_int16 = img_int16.astype(numpy.int32)
            img_int16 = img_int16 << shift
            img_int16 = (img_int16 + (maxint + 1)) % (2 * (maxint + 1)) - maxint - 1
            img_int16 = img_int16 >> shift
            img_int16 = (img_int16 + (maxint + 1)) % (2 * (maxint + 1)) - maxint - 1
        #import cv2
        #cv2.imwrite('/home/fangyunfeng/test_gdcm/%s.png'%os.path.basename(self.dcm_path).split('.')[0], img_int16)
        return MError(MError.E_FIELD_DICOM, 0, 'No error.\n'), img_int16


class SERIES(object):
    '''
    initialize SERIES4CT object，choose series_path or dcm_files
    :series_path：the path of serise
    :dcm_files：files list
    :renturn：NULL
    '''
    def __init__(self, series_path=None, dcm_files=None, strict_check_dicom = True):
        self.err_list = []
        self.warning_list = []
        #series
        self.is_valid_ = True
        self.within_max_spacing = True
        self.thickness_less_spacing = True
        self.thickness_2timesbigthan_spacing = True
        self.instance_dict = {}
        self.dcm_files = []
        self.slice_number = None
        self.ordered = []

        #dcm
        self.patient_id = None
        self.study_uid = None
        self.series_uid = None
        self.slice_spacing = None
        self.slice_thickness = None
        self.pixel_spacing = None
        self.image_loc = None
        self.window_center = None
        self.window_width = None

        self.ctdivol = None
        self.body_part = None
        self.manufacturer = None
        self.kernel = None
        self.kvp = None

        self.img_height = None
        self.img_width = None
        self.series_uid = None
        self.image_position = None
        self.patient_position = None
        self.exten_list = ['.dcm', '.im', '.ima', '.IM', '.IMA', '.DCM']

        self.img_tensor_int16 = None

        if not (series_path or dcm_files):
            self.err_list.append(MError(MError.E_FIELD_SERIES, 'E_SERIES_001', '[Error] SERIES init error, input is empty.\n'))
            self.is_valid_ = False
            return

        if series_path:
            self.series_path = series_path
            for root, dirs, files in os.walk(self.series_path):
                for file_i in files:
                    exten_str = os.path.splitext(file_i)[1]
                    if exten_str in self.exten_list:
                        self.dcm_files.append(os.path.join(root, file_i))
                break
            if len(self.dcm_files) == 0:
                self.dcm_files = glob.glob(series_path + '/*')
        if dcm_files:
            self.dcm_files = dcm_files
        if len(self.dcm_files) == 0:
            self.err_list.append(MError(MError.E_FIELD_SERIES, 'E_SERIES_002', '[Error] SERIES files number = 0 .\n'))
            self.is_valid_ = False
            return

        self.series = []
        self.slice_number = len(self.dcm_files)
        '''Initialize with the first valid file'''
        for dcm_file in self.dcm_files:
            if not os.path.isfile(dcm_file):
                continue
            dcm_obj = DICOM(dcm_file)
            if not dcm_obj.is_valid():
                continue
            self.patient_id = dcm_obj.pid  # patient ID
            self.study_uid = dcm_obj.study_uid  # the check time ID
            self.pixel_spacing = dcm_obj.pixel_spacing  # the interval between the pixel 
            self.image_loc = dcm_obj.image_loc  # image location 
            self.window_center = dcm_obj.window_center
            self.window_width = dcm_obj.window_width
            self.ctdivol = dcm_obj.ctdivol
            self.body_part = dcm_obj.body_part
            self.manufacturer = dcm_obj.manufacturer
            self.kernel = dcm_obj.kernel
            self.kvp = dcm_obj.kvp
            self.img_height = dcm_obj.img_height
            self.img_width = dcm_obj.img_width
            self.series_uid = dcm_obj.series_uid  # series number 
            self.image_position = dcm_obj.image_position
            self.patient_position = dcm_obj.patient_position
            self.slice_thickness = dcm_obj.slice_thickness
            self.flip = False 
            break

        for dcm_file in self.dcm_files:
            '''
            check whether the field in each dcm is valid
            '''
            if not os.path.isfile(dcm_file):
                continue
            dcm_obj = DICOM(dcm_file)
            if not dcm_obj.is_valid():
                if strict_check_dicom:
                    self.is_valid_ = False
                    for err in dcm_obj.err_list:
                        self.err_list.append(err)
                    for war in dcm_obj.warning_list:
                        self.warning_list.append(war)
                    return
                else:
                    for dcm_war in dcm_obj.warning_list:
                        b_append = True
                        for s_war in self.warning_list:
                            if dcm_war.e_no == s_war.e_no:
                                b_append = False
                        if b_append:
                            self.warning_list.append(dcm_war)
                    continue
            if dcm_obj.img_height != self.img_height or dcm_obj.img_width != self.img_width: 
                self.is_valid_ = False
                self.err_list.append(MError(MError.E_FIELD_SERIES, 'E_SERIES_003', '[Error] SERIES with different shape，dcm_file:%s.\n' %dcm_file))
                return
            self.series.append(dcm_obj)
            self.ordered.append([dcm_obj, dcm_file])

        if len(self.series) < 3:
            self.is_valid_ = False
            self.err_list.append(MError(MError.E_FIELD_SERIES, 'E_SERIES_004', '[Error] SERIES error valid dicom_num < 3，dcm_files:%s.\n' % self.dcm_files))
            return

        self.series = sorted(self.series, key=lambda x: x.instance_no)
        self.ordered = sorted(self.ordered, key=lambda x: x[0].instance_no)
        self.origin_point = self.series[0].image_loc
        self.end_point = self.series[-1].image_loc
        self.phyics_direct = numpy.array([1, 1, numpy.sign(self.series[-1].image_loc[2] - self.series[0].image_loc[2])])
        obj0, obj1 = self.series[1:3]
        num_slices01 = obj0.instance_no - obj1.instance_no
        if (obj0.image_loc is not None) and (obj1.image_loc is not None):
            if (len(list(obj0.image_loc)) > 2) and (len(list(obj1.image_loc)) > 2):
                slice_loc01 = obj0.image_loc[2] - obj1.image_loc[2]
            else:
                slice_loc01 = 0
        else:
            slice_loc01 = 0
        if slice_loc01 * num_slices01 > 0:
            self.flip = True
        if num_slices01 == 0:
            self.is_valid_ = False
            self.err_list.append(MError(MError.E_FIELD_SERIES, 'E_SERIES_005', '[Error] Instance_no is invalid，dcm_files:%s.\n' %self.dcm_files[0]))
            return
        self.slice_spacing = round(abs(float(obj0.slice_loc - obj1.slice_loc) / float(num_slices01)), 5)
        if self.slice_spacing <= 0.001:
            self.is_valid_ = False
            self.err_list.append(MError(MError.E_FIELD_SERIES, 'E_SERIES_006', '[Error] Slice_spacing must be positive，dcm_files:%s(slice_loc:%f) and %s(slice_loc:%f).\n' %(
                obj0.dcm_path, obj0.slice_loc, obj1.dcm_path, obj1.slice_loc)))
            return

    '''
    :return：whether valid。True/False
    '''
    def is_valid(self):
        return self.is_valid_

    '''
    check the series according to the limit value。
    :min_dicom_num：the minimum of dcm files dcm_files < min_dicom_num
    :max_slice_thickness：the max slice thickness，slice_thickness > max_slice_thickness
    :max_slice_spacing：max slice spacing，slice_spacing > max_slice_spacing
    :check_continue：check the dcm in series is continue，default True
    :check_unique：default true
    :check_spacing：default true slice_thickness >= slice_spacing and slice_thickness / slice_spacing > 2 (pitch)
    :return：MError, is_valid_
    '''
    def gen_series(self, min_dicom_num = 1, max_slice_thickness = 30, max_slice_spacing = 30, check_continue = True):

        if not self.is_valid():
            if len(self.err_list) > 0:
                return self.err_list[0], False
            else:
                err = MError(MError.E_FIELD_SERIES, 'E_SERIES_001', '[Error] SERIES is not valid （gen_series）.\n')
                self.err_list.append(err)
                return err, False

        '''dcm file num  >=  min_dicom_num'''
        if len(self.dcm_files) < min_dicom_num:
            self.is_valid_ = False
            err = MError(MError.E_FIELD_SERIES, 'E_SERIES_010',
                   '[Error] DICOM files are not enough (at least %d, but only %d).\n' % (
                   min_dicom_num, len(self.dcm_files)))
            self.err_list.append(err)
            return err, self.is_valid_

        '''thickness >= max_slice_thickness'''
        if self.slice_thickness > max_slice_thickness:
            self.is_valid_ = False
            err = MError(MError.E_FIELD_SERIES, 'E_SERIES_011',
                                       '[Error] Slice thickness is not suported (max: %f mm, input: %f mm).\n' % (max_slice_thickness, self.slice_thickness))
            self.err_list.append(err)
            return err, self.is_valid_
        '''spacing >= max_slice_spacing'''
        if self.slice_spacing > max_slice_spacing:
            self.is_valid_ = False
            err = MError(MError.E_FIELD_SERIES, 'E_SERIES_012',
                                       '[Error] Slice spacing is not suported(max:%f mm, input:%f mm).\n' % (
                                           max_slice_spacing, self.slice_spacing))
            self.err_list.append(err)
            return err, self.is_valid_

        dcm_slice_thickness_list = []
        dcm_slice_spacing_list = []
        warning_str = ''
        self.img_tensor_int16 = numpy.zeros([self.slice_number, self.img_height, self.img_width], dtype=numpy.int16)
        for idx in range(1, len(self.series)):
            dcm_obj = self.series[idx]
            self.instance_dict[idx] = self.series[idx].instance_no
            if dcm_obj.study_uid != self.study_uid:
                self.is_valid_ = False
                err = MError(MError.E_FIELD_SERIES, 'E_SERIES_013',
                                           '[Error] Study uid inconformity：%s and %s.\n' % (
                                               str(dcm_obj.study_uid), str(self.study_uid)))
                self.err_list.append(err)
                return err, self.is_valid_
            if dcm_obj.series_uid != self.series_uid:
                self.is_valid_ = False
                err = MError(MError.E_FIELD_SERIES, 'E_SERIES_014',
                                           '[Error] Series_uid inconformity：%s and %s.\n' % (
                                               str(dcm_obj.series_uid), str(self.series_uid)))
                self.err_list.append(err)
                return err, self.is_valid_
            if dcm_obj.pid != self.patient_id:
                self.is_valid_ = False
                err = MError(MError.E_FIELD_SERIES, 'E_SERIES_015',
                                           '[Error] Patient_id inconformity：%s and %s.\n' % (
                                               str(dcm_obj.pid), str(self.patient_id)))
                self.err_list.append(err)
                return err, self.is_valid_

            dcm_obj_prev = self.series[idx - 1]
            num_slices = dcm_obj.instance_no - dcm_obj_prev.instance_no
            if check_continue:
                if num_slices != 1:
                    err = MError(MError.E_FIELD_SERIES, 'E_SERIES_007',
                           '[Error] Instance number is discontinue，dcm_files:%s (%f) and dcm_file:%s (%f).\n' % (
                               dcm_obj.dcm_path, dcm_obj.instance_no,
                               dcm_obj_prev.dcm_path,
                               dcm_obj_prev.instance_no))
                    self.err_list.append(err)
                    self.is_valid_ = False
                    return err, self.is_valid_

            slice_spacing = round(abs(float(dcm_obj_prev.slice_loc - dcm_obj.slice_loc) / float(num_slices)), 5)
            if slice_spacing <= 0.001:
                self.is_valid_ = False
                err = MError(MError.E_FIELD_SERIES, 'E_SERIES_006',
                                            '[Error] Slice_spacing must be positive，dcm_files:%s(slice_loc:%f) and %s(slice_loc:%f).\n' % (
                                                dcm_obj_prev.dcm_path, dcm_obj_prev.slice_loc,
                                                dcm_obj.dcm_path, dcm_obj.slice_loc))
                self.err_list.append(err)
                return err, self.is_valid_

            if abs(self.slice_spacing - slice_spacing) > 0.01:
                warning_str = 'warning.slice_spacing_not_unique.\n'

            if abs(self.slice_thickness - dcm_obj.slice_thickness) > 0.01:
                warning_str = '[Warning] warning.slice_thickness_not_consistent.\n'

            dcm_slice_thickness_list.append(dcm_obj.slice_thickness)
            dcm_slice_spacing_list.append(slice_spacing)

            if idx == 1:
                self.instance_dict[0] = self.series[0].instance_no
                status, img = self.series[0].get_image()
                if status.check():
                    self.img_tensor_int16[0, :, :] = img[:, :]
                else:
                    self.warning_list.append(MError(MError.E_FIELD_SERIES, 'W_SERIES_001',
                                                    "get_image error ,dcm:%s"%self.series[0].instance_no))
            status, img = dcm_obj.get_image()
            if status.check():
                self.img_tensor_int16[idx, :, :] = img[:, :]
            else:
                self.warning_list.append(MError(MError.E_FIELD_SERIES, 'W_SERIES_001',
                                                "get_image error ,dcm:%s" % dcm_obj.instance_no))

        self.img_tensor_int16 = numpy.array(self.img_tensor_int16)
        if warning_str != '':  
            self.warning_list.append(MError(MError.E_FIELD_SERIES, 'W_SERIES_001', warning_str))
            self.slice_thickness = stats.mode(numpy.array(dcm_slice_thickness_list))[0][0]
            self.slice_spacing = stats.mode(numpy.array(dcm_slice_spacing_list))[0][0]

        '''slice_thickness >= slice_spacing'''
        if  self.slice_thickness < self.slice_spacing:
            self.warning_list.append(MError(MError.E_FIELD_DICOM, 'W_SERIES_002',
                                       '[Warning] DICOM Warning slice_thickness_less_than_spacing.\n'))

        return MError(MError.E_FIELD_SERIES, 0, 'No error.\n'), self.is_valid_


