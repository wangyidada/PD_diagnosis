import os
import numpy as np
import random
from utils.my_utils import load_data
import sys
sys.path.append('/homes/ydwang/projects')


def split_dataset_folds(data_dir, test_index_percent=0.2, nfold=4, seed=42):
    test_num = np.floor(len(data_dir)*test_index_percent)
    np.random.seed(seed)
    test_patients_list = random.sample(data_dir, test_num)
    tv_patients_list = [x for x in data_dir if x not in test_patients_list]

    n_patients = len(tv_patients_list)
    pid_idx = np.arange(n_patients)
    np.random.seed(seed)
    np.random.shuffle(pid_idx)
    n_fold_list = np.array_split(pid_idx, nfold)

    val_patients_list = []
    train_patients_list = []
    train_fold = []
    val_fold = []
    for j in range(nfold):
        for i, fold in enumerate(n_fold_list):
            if i == j:
                for idx in fold:
                    val_patients_list.append(tv_patients_list[idx])
            else:
                for idx in fold:
                    train_patients_list.append(tv_patients_list[idx])
        train_fold.append(train_patients_list)
        val_fold.append(val_patients_list)
    return train_fold, val_fold, test_patients_list


def split_dataset(files, percentage=[0.65, 0.15, 0.2], number=[], modes='percent', random_seed=42, save_path=None):
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    # files = os.listdir(folder_path)
    random.shuffle(files)
    if modes == 'percent':
        train_num = np.floor(len(files)*percentage[0])
        val_num = np.floor(len(files)*percentage[1])
        test_num = len(files) - train_num - val_num
        number = [train_num, val_num, test_num]
        number = [np.int(x) for x in number]
    else:
        number = number
    random.seed(random_seed)
    train_patients_list = random.sample(files, number[0])
    vt_list = [x for x in files if x not in train_patients_list]
    random.seed(random_seed)
    val_patients_list = random.sample(vt_list, number[1])
    test_patients_list = [x for x in vt_list if x not in val_patients_list]
    print(f"train patients: {number[0]}, val patients: {number[1]}, test patients: {number[2]}")

    if save_path != None:
        np.save(os.path.join(save_path, 'train_index.npy'), train_patients_list)
        np.save(os.path.join(save_path, 'val_index.npy'), val_patients_list)
        np.save(os.path.join(save_path, 'test_index.npy'), test_patients_list)
    return [train_patients_list, val_patients_list, test_patients_list]


def find_roi_slice(roi_data):
    '''
    the shape of roi_data is S, H, W
    '''
    min_index, max_index = 0, roi_data.shape[0] - 1
    all_slice = list(range(roi_data.shape[0]))
    all_slice.remove(min_index)
    all_slice.remove(max_index)

    roi_sum = np.sum(roi_data, axis=(1, 2))
    with_roi = list(np.where(roi_sum != 0)[0])

    if min_index in with_roi:
        with_roi.remove(min_index)
    if max_index in with_roi:
        with_roi.remove(max_index)
    without_roi = list(set(all_slice) - set(with_roi))
    print(roi_data.shape, with_roi, without_roi)
    # from MeDIT.Visualization import Imshow3DArray
    # Imshow3DArray(np.transpose(roi_data, [1, 2, 0]))
    return with_roi, without_roi


def data2dict(folder_path, index_path, roi_mode, save_name):
    roi_dict = {}
    index = np.load(index_path).tolist()
    for case in index:
        case_files = os.listdir(os.path.join(folder_path, case))
        roi_name = [x for x in case_files if roi_mode in x]
        if len(roi_name) == 0:
            continue
        else:
            roi_path = os.path.join(folder_path, case, roi_name[0])
            roi_data = load_data(roi_path)
            with_roi, without_roi = find_roi_slice(roi_data)
            roi_dict[case] = [with_roi, without_roi]
    np.save(save_name, roi_dict)


if __name__ == '__main__':
    folder_path = r'/home/wyd/PycharmProjects/RJH/RJ_DATA_Seg_0427/stage'
    save_path = r'/home/wyd/PycharmProjects/RJH/RJ_DATA_Seg_0427/index'
    os.makedirs(save_path, exist_ok=True)
    dataset = split_dataset(folder_path, save_path=save_path, percentage=[0.6, 0.15, 0.25])

    # os.makedirs(os.path.join(save_path, 'dict_folder'), exist_ok=True)
    # for dataset in ['train', 'val']:
    #     data2dict(folder_path, os.path.join(save_path, 'index', dataset + '_index.npy'),
    #               roi_mode='label_N.npy', save_name=os.path.join(save_path, 'dict_folder', dataset + '_index.npy'))

    train_csv_file =  r'/homes/ydwang/projects/RJ_PD_dignosis/index/without_up/train_index.csv'
    eval_csv_file =  r'/homes/ydwang/projects/RJ_PD_dignosis/index/without_up/val_index.csv'
    save_folder = r'/homes/ydwang/projects/RJ_PD_dignosis/index/5_fold'
    for i in range(5):
        save_path = os.path.join(save_folder, 'fold_' + str(i))
        split_dataset_folds(data_dir,  nfold=5, seed=42, select=i, save_path=save_path)






