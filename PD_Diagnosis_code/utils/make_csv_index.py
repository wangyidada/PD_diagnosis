import sys
sys.path.append('/homes/ydwang/projects')
from utils.Split_Dataset import split_dataset_folds
import csv
import os
import pandas as pd
import numpy as np


class make_index_csv:
    def __init__(self, case_list, save_csv_path, test_index_percent=0.2, nfold=4, is_upsample=False, up_label=1, up_times=0):
        self.case_list = case_list
        self.save_path = save_csv_path
        self.test_index_percent = test_index_percent
        self.nfold = nfold
        self.is_upsample = is_upsample
        self.up_times = up_times
        self.up_label = up_label
        self.csv_names = ['train_index.csv', 'val_index.csv', 'test_index.csv']

    def get_csv(self):
        for m in range(self.nfold):
            train_list, val_list, test_list = [], [], []
            train_label_list, val_label_list, test_label_list = [], [], []

            for label, case in enumerate(self.case_list):
                [train_nfold, val_nfold, test_index] = split_dataset_folds(case, self.test_index_percent, self.nfold)
                train_index = train_nfold[m]
                val_index = val_nfold[m]
                if self.is_upsample == True and label == self.up_label:
                    for i in range(self.up_times):
                        train_list += train_index
                        train_label_list += len(train_index) *[label]
                        val_list += val_index
                        val_label_list += len(val_index) *[label]
                    test_list += test_index
                    test_label_list += len(test_index) *[label]
                else:
                    train_list += train_index
                    train_label_list += len(train_index) *[label]
                    val_list += val_index
                    val_label_list += len(val_index) *[label]
                    test_list += test_index
                    test_label_list += len(test_index) *[label]

            index_list = [train_list, val_list, test_list]
            label_list = [train_label_list, val_label_list, test_label_list]
            print(test_list)

            for i in range(len(self.csv_names)):
                row_list =[]
                save_folder = os.path.join(self.save_path, 'fold_' + str(m))
                os.makedirs(save_folder, exist_ok=True)
                with open(os.path.join(save_folder, self.csv_names[i]), 'w') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(['ID', 'label'])
                    for j in range(len(index_list[i])):
                        row_list.append([index_list[i][j], label_list[i][j]])
                    writer.writerows(row_list)


def get_case_list(folder, csv_path):
    PD_list, NC_list = [], []
    df = pd.read_excel(csv_path)
    data = np.asarray(df.values)[:, 0:2]
    files = os.listdir(folder)
    other_list = []
    print(len(files))
    for file in files:
        for i in range(data.shape[0]):
            id = data[i][1]
            if str(id) == str(file[0:10]):
                label = data[i][0]
                if label == 'PD':
                    PD_list.append(file)
                elif label == 'NC':
                    NC_list.append(file)
                elif label == 'NC ':
                    NC_list.append(file)
                else:
                    other_list.append(file)
    print(len(NC_list), len(PD_list))
    return PD_list, NC_list


if __name__ == '__main__':
    csv_file = r'/homes/ydwang/Data/rujin_case_list_key-20200530-toECNU.xlsx'
    folder = r'/homes/ydwang/Data/stage_data_0616'
    [PD_list, NC_list] = get_case_list(folder, csv_file)

    save_path = r'/homes/ydwang/projects/RJ_PD_dignosis/index/5_fold_up'
    make_index_csv(case_list=[NC_list, PD_list], save_csv_path=save_path,nfold=5,
                   is_upsample=True,  up_label=1, up_times=3).get_csv()
