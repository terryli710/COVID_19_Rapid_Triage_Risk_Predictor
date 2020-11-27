import os
import os.path as osp
import cv2
import time
import csv
import numpy as np

def get_lesion_mask(lesion_mask_path):
    '''
    read all png file in lesion mask path to generate the lesion mask
    '''
    lesion_file_list = os.listdir(lesion_mask_path)
    png_list = []
    for lesion_file in lesion_file_list:
        if lesion_file.endswith("png"):
            png_list.append(lesion_file)
    png_list = sorted(png_list)
    png_file = osp.join(lesion_mask_path, png_list[0])
    png_data = cv2.imread(png_file, cv2.IMREAD_UNCHANGED)
    lesion_mask = np.zeros([len(png_list), png_data.shape[0], png_data.shape[1]], dtype=np.uint8)
    for idx, png_name in enumerate(png_list):
        png_file = osp.join(lesion_mask_path, png_list[idx]) 
        png_data = cv2.imread(png_file, cv2.IMREAD_UNCHANGED)
        lesion_mask[idx, :, :] = png_data[:,:]
    return lesion_mask

def mkdir_safe(d):
    '''
    generate the target diretory safety
    '''
    if type(d) not in [str]:
        print('param type error: d is %s, expected to be str' % type(d).__name__)
    sub_dirs = d.split('/|\\')
    d_split = d.split('/')
    sub_dirs = []
    for x in d_split:
        sub_dirs = sub_dirs + x.split("\\")
    cur_dir = ''
    max_check_times = 5
    sleep_seconds_per_check = 0.001
    for i in range(len(sub_dirs)):
        cur_dir += sub_dirs[i] + '/'
        for check_iter in range(max_check_times):
            if not os.path.exists(cur_dir):
                try:
                    os.mkdir(cur_dir)
                except:
                    time.sleep(sleep_seconds_per_check)
                    continue
            else:
                break
    if not osp.exists(d):
        print("permission denied")

def save_to_csv(csv_file, feature_vector):
    '''
    save the extracted feature to csv file
    '''
    with open(csv_file, 'w') as csv_f:
        writer = csv.writer(csv_f, lineterminator='\n')
        headers = None
        if headers is None:
            headers = list(feature_vector.keys())
        writer.writerow(headers)
        row = []
        for h in headers:
            row.append(feature_vector.get(h, 'N/A'))
        writer.writerow(row)

def iterate_files(root, include_post=(), exclude_post=()):
    """
    recurrently find files
    :param root:
    :param include_post: default=(), the list of the post of included file names, e.g., ['.dcm']
    :param exclude_post: default=(), the list of the post of excluded file names, e.g., ['.txt']
    :return:
    """
    assert osp.isdir(root),'%s is not a directory' % root
    result = []
    for root,dirs,files in os.walk(root, topdown=True, followlinks=True):
        for fl in files:
            if len(include_post) != 0:
                if osp.splitext(fl)[-1] == include_post:
                    result.append(os.path.join(root,fl))
            else:
                if osp.splitext(fl)[-1] not in exclude_post:
                    result.append(os.path.join(root, fl))
    return result

def generate_subdirs(dicom_dir, include_post):
    '''
    generate the all subdir in one dicom_dir
    '''
    all_files = iterate_files(dicom_dir, include_post) 
    sub_dir_file_dict = {}
    for file_path in all_files:
        assert osp.isfile(file_path)
        items = osp.dirname(file_path).split('/')
        N = len(dicom_dir.split('/')) - 1
        sub_dir = '/'.join(items[N:])
        if sub_dir not in sub_dir_file_dict:
            sub_dir_file_dict[sub_dir] = []
        sub_dir_file_dict[sub_dir].append(file_path)
    new_sub_dir_file_dict = {}
    for sub_dir in sub_dir_file_dict:
        if len(sub_dir_file_dict[sub_dir]) < 1:
            continue
        new_sub_dir_file_dict[sub_dir] = sub_dir_file_dict[sub_dir]
    subdir_list = sorted(new_sub_dir_file_dict.keys())
    return subdir_list


def valid_csv_file(csv_file, feature_num):
    '''
    check whether the csv file is valid
    '''
    if (not osp.exists(csv_file)) or osp.getsize(csv_file) == 0:
        return False
    if feature_num is not None:
        with open(csv_file, 'r') as csv_f:
            reader = csv.reader(csv_f)
            if len(list(reader)[1]) != feature_num:
                return False
    return True

def get_feature_num_from_save_csv(csv_root, sub_dir_list):
    '''
    check all csv file to decide the extracted feature number
    '''
    feature_num_list = []
    for idx, sub_dir in enumerate(sub_dir_list):
        csv_file_list = os.listdir(osp.join(csv_root, sub_dir))
        # csv_file_list = sorted(csv_file_list, key=lambda x:int(x[:-4]))
        for csv_file in csv_file_list:
            csv_file_name = osp.join(csv_root, sub_dir, csv_file)
            with open(csv_file_name, 'r') as csv_f:
                reader = csv.reader(csv_f)
                feature_num = list(reader)[1]
                feature_num_list.append(len(feature_num))

    if len(feature_num_list) != 0:
        check_feature_num = max(feature_num_list)
        return check_feature_num
    else:
        return None

def merge_csv(csv_root, sub_dir_list, sav_csv_file):
    '''
    merge the csv file in assgined subdir to get the total csv file
    '''
    with open(sav_csv_file, 'w') as sav_f:
        writer = csv.writer(sav_f, lineterminator='\n')
        csv_file_list = os.listdir(osp.join(csv_root, sub_dir_list[0]))
        csv_file_name = osp.join(csv_root, sub_dir_list[0], csv_file_list[0])
        with open(csv_file_name, 'r') as csv_f:
            reader = csv.reader(csv_f)
            header = list(reader)[0]
            writer.writerow(header)
        for idx, sub_dir in enumerate(sub_dir_list):
            csv_file_list = os.listdir(osp.join(csv_root, sub_dir))
            csv_file_list = sorted(csv_file_list, key=lambda x:int(x[:-4]))
            for csv_file in csv_file_list:
                csv_file_name = osp.join(csv_root, sub_dir, csv_file)
                with open(csv_file_name, 'r') as csv_f:
                    reader = csv.reader(csv_f)
                    for idx, item in enumerate(reader):
                        if idx != 0:
                            writer.writerow(item)


# def debug():
#     iterate_files('.\\data\\lesion_mask_png\\', '.png')

# if __name__ == '__main__':
#     debug()