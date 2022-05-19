import numpy as np
import csv
import os
import pandas as pd
import torch
from utils.Result_analysis_cv import classificaton_cv_evaluation, classificaton_single_evaluation


def ensemble_preds(value, save_folder):
    name_list = []
    pred_mean_list = []
    for z in range(np.shape(value)[0]):
        name = value[z, 0]
        p = value[z, 2:]
        label = value[z, 1]
        m = np.mean(p)
        pred_mean_list.append((name, label, m))
        name_list.append(name)

    tv_name_list = set(name_list)
    prediction_list = []
    for x in tv_name_list:
        a = []
        b = []
        for y in pred_mean_list:
            if y[0] == x:
                a.append(y[2])
                b.append(y[1])
        pred_case = np.mean(a)
        label_case = np.mean(b)
        prediction_list.append((x, label_case, pred_case))

    with open(save_folder, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'label', 'pred'])
        writer.writerows(prediction_list)


def single_eval_test(csv_file):
    df = pd.read_csv(csv_file)
    target = df['label'].values
    pred = df['logits'].values
    target = list(target)
    pred = list(pred)
    classificaton_single_evaluation([[target, pred]])


def ensemble_results(path, fode, save_folder, is_test_set=False):
    test_csv_list = []
    tv_csv_list = []
    if is_test_set:
        for i in range(fode):
            folder = 'fode_' + str(i)
            test_csv = pd.read_csv(os.path.join(path, folder, 'test_result.csv'))
            test_csv.rename(columns={'pred': 'pred' + str(i)}, inplace=True)
            test_csv_list.append(test_csv)
        test_pd = test_csv_list[0]
        for j in range(fode - 1):
            test_pd = pd.merge(test_pd, test_csv_list[j + 1], on=['ID', 'label'])
        test_value = test_pd.values
        ensemble_preds(test_value, os.path.join(save_folder, 'test_ensemble.csv'))
    else:
        for i in range(fode):
            folder = 'fode_' + str(i)
            train_csv = pd.read_csv(os.path.join(path, folder, 'train_result.csv'))
            val_csv = pd.read_csv(os.path.join(path, folder, 'val_result.csv'))
            test_csv = pd.read_csv(os.path.join(path, folder, 'test_result.csv'))
            test_csv.rename(columns={'pred': 'pred' + str(i)}, inplace=True)
            tv_csv = pd.concat([train_csv, val_csv], axis=0, ignore_index=True)
            tv_csv.rename(columns={'pred': 'pred' + str(i)}, inplace=True)

            tv_csv_list.append(tv_csv)
            test_csv_list.append(test_csv)
        tv_pd = tv_csv_list[0]
        test_pd = test_csv_list[0]
        for j in range(fode-1):
            tv_pd = pd.merge(tv_pd, tv_csv_list[j+1], on=['ID', 'label'])
            test_pd = pd.merge(test_pd, test_csv_list[j+1], on=['ID', 'label'])

        tv_value = tv_pd.values
        test_value = test_pd.values

        ensemble_preds(tv_value, os.path.join(save_folder, 'tv_ensemble.csv'))
        ensemble_preds(test_value, os.path.join(save_folder, 'test_ensemble.csv'))


def plot_cv_roc(csv_folder, is_test_set=False):
    if is_test_set:
        test_df = pd.read_csv(os.path.join(csv_folder, 'test_ensemble.csv'))
        test_logits = np.asarray(test_df['pred'].values, dtype=np.float32)
        test_label = test_df['label'].values
        test_label = torch.nn.functional.one_hot(torch.from_numpy(test_label).long())
        test_label = np.asarray(test_label.numpy()[:, 1], dtype=np.int32)
        classificaton_single_evaluation([[test_label, test_logits]], save_fig_file=csv_folder)
    else:
            tv_df = pd.read_csv(os.path.join(csv_folder, 'tv_ensemble.csv'))
            test_df = pd.read_csv(os.path.join(csv_folder, 'test_ensemble.csv'))

            tv_logits = np.asarray(tv_df['pred'].values, dtype=np.float32)
            tv_label = tv_df['label'].values
            tv_label = torch.nn.functional.one_hot(torch.from_numpy(tv_label).long())
            tv_label = np.asarray(tv_label.numpy()[:, 1], dtype=np.int32)

            test_logits = np.asarray(test_df['pred'].values, dtype=np.float32)
            test_label = test_df['label'].values
            test_label = torch.nn.functional.one_hot(torch.from_numpy(test_label).long())
            test_label = np.asarray(test_label.numpy()[:, 1], dtype=np.int32)

            classificaton_cv_evaluation([[tv_label, tv_logits], [test_label, test_logits]],  save_fig_file=csv_folder)

