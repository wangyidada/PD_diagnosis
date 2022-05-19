# from MeDIT.Visualization import Imshow3DArray
import sys
sys.path.append('/homes/ydwang/projects')
import numpy as np
import os
import SimpleITK as sitk
import random
import math
from MeDIT.SaveAndLoad import LoadNiiData
from skimage import exposure


def show_data(data, roi_list):
    data = np.transpose(data, [1, 2, 0])
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    roi_list = [np.transpose(x, [1, 2, 0]) for x in roi_list]
    roi_list = [np.asarray(x, dtype=np.float32) for x in roi_list]
    # Imshow3DArray(data, roi_list)


def Dice(input, target):
    """soft dice loss"""
    eps = 1e-7
    iflat = input.reshape(-1)
    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()
    return 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)


def find_file_path(path, key_word_list):
    files = os.listdir(path)
    select_files_path = []
    for key_word in key_word_list:
        file = [x for x in files if key_word in x]
        if len(file) == 0:
            print('No file:', path, key_word)
        elif len(file) == 1:
            file_path = os.path.join(path, file[0])
            select_files_path.append(file_path)
        else:
            print('More than One file', path, key_word)
    return select_files_path


def load_data(data_path):
    if os.path.splitext(data_path)[-1] == '.nii':
        _, data, _ = LoadNiiData(data_path)
    elif os.path.splitext(data_path)[-1] == '.gz':
        _, data, _ = LoadNiiData(data_path)
    elif os.path.splitext(data_path)[-1] == '.npy':
        data = np.load(data_path)
    else:
        print(data_path, 'wrong roi file!!!!!')
        data = None
    return data


def Nii2Numpy(file_path):
    image = sitk.ReadImage(file_path)
    data = sitk.GetArrayFromImage(image)
    return data


def getDataList(dict_path, rate=0.8):
    data_list = []
    data_dict = np.load(dict_path, allow_pickle=True).item()
    for name in data_dict.keys():
        with_roi_index = data_dict[name][0]
        without_roi_index = data_dict[name][1]
        without_roi_num = int(math.floor(len(without_roi_index) * rate))
        select_without_roi = random.sample(without_roi_index, without_roi_num)
        select_results = with_roi_index + select_without_roi
        for s in select_results:
            data_list.append((name, s))
    return data_list


def get_test_data_list(folder_path, case, data_mode, is_3slice=True):
    data_list = []
    case_files = os.listdir(os.path.join(folder_path, case))
    data_name = [x for x in case_files if data_mode[0] in x]
    assert len(data_name) <= 1, "wong roi_mode"
    data_path = os.path.join(folder_path, case, data_name[0])
    data = load_data(data_path)
    s = data.shape[0]
    if is_3slice:
        for i in range(1, s - 1):
            data_list.append((case, i))
    else:
        for i in range(s):
            data_list.append((case, i))
    return data_list


def normalize(img):
    mn = img.min()
    mx = img.max()
    img = (img - mn) / (mx - mn)
    return img


def standardization(img):
    mean = img.mean()
    std = img.std()
    denominator = np.reciprocal(std)
    img = (img - mean) * denominator
    return img


def clahe_img(img):
    img = normalize(img)
    clahe_img = exposure.equalize_adapthist(img)
    return clahe_img


def GetIndexRangeInROI(roi_mask, target_value=1, find_type=0):
    '''
    find_type denotes how to determine one ROI:
    0 denotes the exact ROI;
    1 denotes the region larger or equal to target_value;
    2 denotes the region smaller or equal to target_value.
    '''
    if find_type == 0:
        if np.ndim(roi_mask) == 2:
            x, y = np.where(roi_mask == target_value)
            x = np.unique(x)
            y = np.unique(y)
            return x.tolist(), y.tolist()
        elif np.ndim(roi_mask) == 3:
            x, y, z = np.where(roi_mask == target_value)
            x = np.unique(x)
            y = np.unique(y)
            z = np.unique(z)
            return x.tolist(), y.tolist(), z.tolist()
    elif find_type == 1:
        if np.ndim(roi_mask) == 2:
            x, y = np.where(roi_mask >= target_value)
            x = np.unique(x)
            y = np.unique(y)
            return x.tolist(), y.tolist()
        elif np.ndim(roi_mask) == 3:
            x, y, z = np.where(roi_mask >= target_value)
            x = np.unique(x)
            y = np.unique(y)
            z = np.unique(z)
            return x.tolist(), y.tolist(), z.tolist()
    elif find_type == 2:
        if np.ndim(roi_mask) == 2:
            x, y = np.where(roi_mask <= target_value)
            x = np.unique(x)
            y = np.unique(y)
            return x.tolist(), y.tolist()
        elif np.ndim(roi_mask) == 3:
            x, y, z = np.where(roi_mask <= target_value)
            x = np.unique(x)
            y = np.unique(y)
            z = np.unique(z)
            return x.tolist(), y.tolist(), z.tolist()
