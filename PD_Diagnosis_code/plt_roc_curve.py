from utils import Result_analysis
import pandas as pd
import os
from utils.plot_cv_csv import plot_cv_roc, ensemble_results


def get_pred_result(csv_file):
    df = pd.read_csv(csv_file)
    target = df['label'].values
    pred = df['logits'].values
    target = list(target)
    pred = list(pred)
    return [target, pred]


def plot_mul_roc(path, csv_list, name_list, save_name):
    data_list = []
    for csv in csv_list:
        csv_file = path + '/' + csv + '/result/ensemble_results/test_ensemble.csv'
        df = pd.read_csv(csv_file)
        target = df['label'].values
        pred = df['pred'].values
        target = list(target)
        pred = list(pred)
        data_list.append([target, pred])
    Result_analysis.plot_roc_curve(data_list, colors=['blue', 'green', 'orange', 'red', 'purple'], names=name_list, save_name=save_name)


def main(csv_path, names):
    result_list =[]
    for name in names:
        csv_file = os.path.join(csv_path, name)
        result_list.append(get_pred_result(csv_file))
    Result_analysis.classificaton_evaluation(result_list)


if __name__ == '__main__':
    # csv_folder_path= r'/homes/ydwang/projects/RJ_PD_dignosis/diff_models/Resnet50/result'
    # csv_folder_path=  r'/homes/ydwang/projects/RJ_PD_dignosis/diff_models/vgg16/result'
    # csv_folder_path = r'/homes/ydwang/projects/RJ_PD_dignosis/diff_models/SE-ResNeXt50/result'
    # csv_folder_path = r'/homes/ydwang/projects/RJ_PD_dignosis/diff_models/single_input_3DSeNet_CLAHE/result'
    # csv_names = ['train_result.csv', 'val_result.csv', 'test_result.csv']
    # fode = 5

    # for i in range(fode):
    #     main(os.path.join(csv_folder_path, 'fode_' + str(i)), csv_names)
    # save_folder = os.path.join(csv_folder_path, 'ensemble_results1')
    # os.makedirs(save_folder, exist_ok=True)
    # ensemble_results(csv_folder_path, fode=fode, save_folder=save_folder)
    # plot_cv_roc(save_folder)

    path = r'/homes/ydwang/projects/RJ_PD_dignosis/diff_models'
    result_list = ['vgg16', 'SE-ResNeXt50',  'Resnet50', 'AG-SE-ResNeXt50','mannul-ROI']
    name_list = ['VGG16', 'ResNet50',  'SE-ResNeXt50', 'AG-SE-ResNeXt50', 'AG-SE-ResNeXt50-ROI']
    save_path =r'/homes/ydwang/projects/RJ_PD_dignosis/diff_models/ROCs-all.jpg'
    plot_mul_roc(path, result_list, name_list, save_name= save_path)




