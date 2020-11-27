import os
import os.path as osp
import numpy as np
import SimpleITK as sitk
import pdb
import json
import nrrd
import csv
import traceback
import shutil
import multiprocessing
import argparse
from radiomics import featureextractor
import cv2
from mitok.utils.mdicom import SERIES
from mitok.image.image import image_tensor_resize
from radiom_utils import get_lesion_mask, mkdir_safe, save_to_csv, generate_subdirs, valid_csv_file
from radiom_utils import get_feature_num_from_save_csv, merge_csv

# the configuration to extract radiomics features, you can change this file according to your experimental setup
yaml_file = 'njjz_all_features.yaml'

def check_csv_file(lesion_mask, subdir, feature_num=None):
    '''
    check whether generate all csv file in one ct
    '''
    csv_dir = osp.join(save_root, 'csv', subdir)
    if not osp.exists(csv_dir):
        return False
    lesion_num = np.max(lesion_mask)
    for i in range(lesion_num):
        label = i + 1
        csv_file = osp.join(save_root, 'csv/', subdir, '{}.csv'.format(label))
        if not valid_csv_file(csv_file, feature_num=feature_num):
            return False
    return True

def compute_radiomics(ct_dicom_path, lesion_mask_path, subdir, feature_num=None):
    '''
    extract single data's radiomics feature 
    param ct_dicom_path: the path of ct which need to extract feature
    param lesion_mask_path: the path of ct's lesion which store as png file
    param subdir: the subdir of the data which store as pid/studyuid/seriesuid
    param feature_num: used to check the extract feature number is valid, default is None
    '''
    lesion_mask = get_lesion_mask(lesion_mask_path)
    if check_csv_file(lesion_mask, subdir, feature_num=feature_num):
        return
    image_nrrd_save_file = osp.join(save_root, 'image', subdir, 'image.nrrd')
    mkdir_safe(osp.dirname(image_nrrd_save_file))
    
    ct_series= SERIES(series_path=ct_dicom_path)
    pixel_spacing = ct_series.pixel_spacing    
    slice_spacing = ct_series.slice_spacing
    spacing = [slice_spacing, pixel_spacing[0], pixel_spacing[1]]
    # resampling params[z, y, x]
    norm_spacing = [1, 0.6, 0.6]
    pid, study_id, series_id = ct_series.patient_id.strip(), ct_series.study_uid, ct_series.series_uid
    if not osp.exists(image_nrrd_save_file):
        ct_series.gen_series()
        int16_image = ct_series.img_tensor_int16
        image_shape = int16_image.shape
        norm_shape = list(map(lambda x:int(x[0] * x[1] / x[2]), zip(image_shape, spacing, norm_spacing)))
        # we use our own resampling method instead of the function provided by pyradiomics
        _, norm_int16_image_tensor = image_tensor_resize(int16_image, size=norm_shape, return_type='int16')
        nrrd.write(image_nrrd_save_file, norm_int16_image_tensor)
        
    lesion_num = np.max(lesion_mask)
    mkdir_safe(osp.join(save_root, 'seg', subdir))
    mkdir_safe(osp.join(save_root, 'csv', subdir))
    for i in range(lesion_num):
        label = i + 1
        seg_nrrd_save_file = osp.join(save_root, 'seg', subdir, '{}.nrrd'.format(label))
        outputFilepath = osp.join(save_root, 'csv/', subdir, '{}.csv'.format(label))
        if valid_csv_file(outputFilepath, feature_num):
            continue
        if not osp.exists(seg_nrrd_save_file):
            mask_seg_tensor = np.zeros(lesion_mask.shape, dtype='uint8')
            mask_seg_tensor[lesion_mask == label] = 1 
            seg_shape = mask_seg_tensor.shape
            norm_seg_shape = list(map(lambda x:int(x[0] * x[1] / x[2]), zip(seg_shape, spacing, norm_spacing)))
            # we use our own resampling method instead of the function provided by pyradiomics
            _, norm_seg_tensor = image_tensor_resize(mask_seg_tensor, size=norm_seg_shape)
            nrrd.write(seg_nrrd_save_file, norm_seg_tensor)
        featureVector = {}
        featureVector['ct_path'] = ct_dicom_path
        featureVector['patient_id'] = pid
        featureVector['study_id'] = study_id
        featureVector['series_id'] = series_id
        try:
            extractor = featureextractor.RadiomicsFeatureExtractor(yaml_file)
            featureVector.update(extractor.execute(image_nrrd_save_file, seg_nrrd_save_file))
            save_to_csv(outputFilepath, featureVector)
        except Exception:
            print('feature extraction failed')

def run_task(pid, sub_dir_list, start, stop, dicom_root, lesion_mask_root, feature_num=None):
    '''
    param pid: process id
    param sub_dir_list: list, all CT paths
    param start: the index of sub_dir_list to start processing
    param stop: the index of sub_dir_list to stop processing
    param dicom_root:
    param lesion_mask_root:
    param feature_num: used to check whether extracted feature is valid, default is None
    '''
    lines = sub_dir_list[start:stop]
    cnt = 0
    for line in lines:
        try:
            ct_dicom_path = osp.join(dicom_root, line)
            lesion_mask_path = osp.join(lesion_mask_root, line)
            compute_radiomics(ct_dicom_path, lesion_mask_path, line, feature_num=feature_num)
        except Exception as ex:
            traceback.print_exc()
            print ("Error in single run, pid = {}!".format(pid), ex) 

def multi_run(dicom_root, lesion_mask_root, save_root):
    '''
    extract features in parallel to save time
    '''
    try:
        # cache path to save the middle and final outputs
        mkdir_safe(osp.join(save_root, 'image'))
        mkdir_safe(osp.join(save_root, 'csv'))
        mkdir_safe(osp.join(save_root, 'seg'))

        # to get all CT paths
        sub_dir_list = generate_subdirs(lesion_mask_root + '/', '.png')
        # the number of CTs to be handled
        N = len(sub_dir_list)
        # if the data is not that so much, use just 1 process
        if N < 10:
            proc_num = 1
        else:
            proc_num = 10 if multiprocessing.cpu_count() > 10 else multiprocessing.cpu_count()

        if proc_num == 1:
            print("extract radiomic feature in single process")
            run_task(0, sub_dir_list, 0, N, dicom_root, lesion_mask_root)
        else: 
            per_part = len(sub_dir_list) // proc_num + 1
            pool = multiprocessing.Pool(processes = proc_num)
            for i in range(proc_num):
                start = i * per_part
                stop = (i + 1) * per_part if (i + 1) * per_part <= N else N
                print('Active Process#{} among {} processes, index from {} to {} of {} CTs.'.format(i + 1, proc_num, start, stop, N))
                pool.apply_async(run_task, (i + 1, sub_dir_list, start, stop, dicom_root, lesion_mask_root))
            pool.close()
            pool.join()
        
        print('Parallel run complete, now re-check the result...')
        check_feature_num = get_feature_num_from_save_csv(osp.join(save_root, 'csv'), sub_dir_list)
        run_task(0, sub_dir_list, 0, N, dicom_root, lesion_mask_root, feature_num=check_feature_num)
        merge_csv_file = osp.join(save_root, 'final_merge_feature.csv')
        merge_csv(osp.join(osp.join(save_root, 'csv')), sub_dir_list, merge_csv_file)
        print('All complete, please open {} to check result.'.format(merge_csv_file))
        print("extracted feature completed, remove the cache")
        shutil.rmtree(osp.join(save_root, 'image'))
        shutil.rmtree(osp.join(save_root, 'csv'))
        shutil.rmtree(osp.join(save_root, 'seg'))
    except Exception as ex:
        traceback.print_exc()
        print ("Error in multi_run extracting radiomics features!", ex)

def parse_opts():
    '''
    set the argument for the project
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dicom_root', default='', type=str, help='the path all ct dicom')
    parser.add_argument('--lesion_mask_root', default='', type=str, help='the path all lesion mask')
    parser.add_argument('--save_root', default='', type=str, help='the path store save file') 
    args = parser.parse_args() 
    return args 

if __name__ == '__main__':
    # opt = parse_opts()
    class opt :
        def __init__(self):
            self.lesion_mask_root = ".\\data\\lesion_mask_png"
            self.dicom_root = ".\\data\\dicom_data"
            self.save_root = ".\\test"
    opt = opt()
    lesion_mask_root = opt.lesion_mask_root
    dicom_root = opt.dicom_root
    save_root = opt.save_root
    multi_run(dicom_root, lesion_mask_root, save_root)



