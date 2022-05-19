import os
from utils.my_utils import load_data
import numpy as np


def get_data(path, data_modes, roi_modes, crop_index, is_multipy_roi=False):
    # the shape of data_list: [[H, W, D], [H, W, D], ......]
    # the shape of roi_list: [[H, W, D]]
    files = os.listdir(path)
    data_list = []
    roi_list = []
    for data_mode in data_modes:
        data_file = [x for x in files if data_mode in x][0]
        data_path = os.path.join(path, data_file)
        data = load_data(data_path)
        data = data[crop_index[0]:crop_index[1], crop_index[2]:crop_index[3] , ...]
        data_list.append(data)

    roi_files = []
    for roi_mode in roi_modes:
        roi_file = [x for x in files if roi_mode in x]
        roi_files = roi_files + roi_file
    if len(roi_files) == 1:
        roi_path = os.path.join(path, roi_files[0])
        roi_data = load_data(roi_path)
    elif len(roi_files) == 2:
        swim_roi_file = [x for x in roi_files if "SWIM" in x][0]
        swim_roi_file_path = os.path.join(path, swim_roi_file)
        swim_roi_data = load_data(swim_roi_file_path)

        T1_roi_file = [x for x in roi_files if "T1" in x][0]
        T1_roi_file_path = os.path.join(path, T1_roi_file)
        T1_roi_data = load_data(T1_roi_file_path)

        # from MeDIT.Visualization import Imshow3DArray
        roi_data = swim_roi_data.copy()
        roi_data[np.where(swim_roi_data == 1)] = 0
        roi_data[np.where(T1_roi_data == 1)] = 1
        # Imshow3DArray(roi_data - swim_roi_data)

    else:
        roi_data =0
        print("Wrong roi files")

    roi_data = roi_data[crop_index[0]:crop_index[1], crop_index[2]:crop_index[3], ...]
    roi_list.append(roi_data)

    if is_multipy_roi == True:
        r = roi_list[0].copy()
        r[np.where(r != 0)] = 1
        data_list = [x * r for x in data_list]

    return data_list, roi_list

