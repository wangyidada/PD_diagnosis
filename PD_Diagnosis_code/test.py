import os
import numpy as np
from Dataset import make_test_loader
import torch
import pandas as pd
from collections import OrderedDict
import time
from models.AG_SE_ResNeXt50 import AG_SE_ResNeXt50_model
from models.SE_ResNeXt import seresnext50
from models.ResNet50 import resnet50
from models.vgg16 import vgg16

def find_best_model(folder_path):
    import re
    best_model_list = []
    folds = os.listdir(folder_path)
    folds = [x for x in folds if 'fold' in x]
    for fold in folds:
        model_folder = os.path.join(folder_path, fold, 'model')
        models = os.listdir(model_folder)
        best_models = [x for x in models if 'best_model' in x]
        epochs=[int(re.findall(r"\d+\.?\d*", y)[0]) for y in best_models]
        best_model =best_models[epochs.index(max(epochs))]
        best_model_list.append(os.path.join(model_folder, best_model))
    return best_model_list


def load_model(model_path, model, is_multi_GPU=False):
    dict = torch.load(model_path)
    network = dict['model']
    if is_multi_GPU:
        new_dict = OrderedDict()
        for k, v in network.items():
            name = k[7:]
            new_dict[name] = v
        model.load_state_dict(new_dict)
    else:
        model.load_state_dict(network)
    return model


def test(loader, model):
    a = time.time()
    model.eval()
    prediction_list = []
    with torch.no_grad():
        for batch_img, batch_label, batch_mask, case in loader:
            batch_img, batch_mask = batch_img.cuda(), batch_mask.cuda()
            # output = model(batch_img, batch_mask)
            output = model(batch_img)
            output = torch.nn.Softmax(dim=1)(output)
            prediction = output.cpu().numpy()
            prediction = np.squeeze(prediction)[1]
            b = time.time()
            # print('time:', b-a)
            label = batch_label.numpy()[0]
            prediction_list.append((case[0], label, prediction))
            print((case[0], label, prediction))
    return prediction_list


def main(params):
    for fode, model_path in enumerate(params['model_path_list']):
        print(model_path)
        model = AG_SE_ResNeXt50_model(num_classes=params['n_classes'])
        # model = resnet50(num_classes=params['n_classes'])
        model = load_model(model_path, model, params['is_multi_GPU'])
        model = model.cuda()

        for set in params['test_file']:
            test_index_file = os.path.join(params["test_csv_file_folder"] + str(fode), set)
            print(test_index_file)
            dataset = make_test_loader(params["data_root"], test_index_file, params["data_modes"], params["roi_modes"],
                                    params["crop_index"], params["input_shape"], batch_size=1)
            prediction = test(dataset, model)
            f = pd.DataFrame(prediction, columns=['ID', 'label', 'logits'])
            save_folder = os.path.join(params['result_path'], 'fode_' + str(fode))
            os.makedirs(save_folder, exist_ok=True)
            save_file = os.path.join(save_folder, set.split('_')[0] + '_result.csv')
            f.to_csv(save_file, index=False)
            print(test_index_file)
            print(save_file)


if __name__ == '__main__':
    AG_SE_ResNeXt50_model = find_best_model('AG_SE/models')
    SE_model = find_best_model('SEnet50/models')
    resnet_model = find_best_model('resnet50/models')
    vgg_model = find_best_model('vgg/models')
    params = {
        'is_multi_GPU': False,
        'fode': 5,
        "data_root": r'/homes/ydwang/Data/data',
        "model_path_list": AG_SE_ResNeXt50_model,
        "test_csv_file_folder": r'/homes/ydwang/projects/RJ_PD_dignosis/index/5_fold_up/fold_',
        "result_path": r'/homes/ydwang/projects/RJ_PD_dignosis/Resnet50/result',
        'test_file': ['train_index.csv', 'val_index.csv', 'test_index.csv'],
        "data_modes": ['QSM.nii'],
        "roi_modes": ['seg.nii.gz'],
        "crop_index": [80, 240, 100, 260],
        "input_shape": [160, 160, 64],
        "n_classes": 2,
    }
    main(params)

















