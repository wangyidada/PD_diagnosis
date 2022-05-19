import numpy as np
from Dataset import make_test_loader
import torch
from collections import OrderedDict
import SimpleITK as sitk
from utils.binary import dc, assd
import csv
from models.network import CA_UNet3D
import os


def load_model(model_path, model):
    dict = torch.load(model_path)
    network = dict['model']
    new_dict = OrderedDict()

    for k, v in network.items():
        name = k[7:]
        new_dict[name] = v
    model.load_state_dict(new_dict)
    return model


def save_sample(seg_volume, original_size, index, store_path):
    seg_volume = torch.nn.functional.softmax(seg_volume, dim=1)
    seg_volume = seg_volume.cpu().numpy()
    seg_volume = np.squeeze(seg_volume)
    seg = np.argmax(seg_volume, axis=0)

    data = np.zeros(original_size)
    data[index[0]:index[1], index[2]:index[3], :] = seg

    image = sitk.GetImageFromArray(data)
    sitk.WriteImage(image, store_path)


def cal_dice_and_hd(seg_volume, reference, c):
    # the shape of seg_volume and reference is [N, C, H, W, D], [N, 1, H, W, D]
    dice_list = []
    hd_list = []
    seg_volume = torch.nn.functional.softmax(seg_volume, dim=1)
    reference = torch.squeeze(reference)
    target = torch.nn.functional.one_hot(reference.long(), num_classes=c)

    seg_volume = seg_volume.cpu().numpy()
    target = target.cpu().numpy()

    for i in range(1, c):
        x = seg_volume[0, i, ...]
        x = x > 0.5
        y = target[..., i]
        x = np.array(x, dtype=np.int)
        y = np.array(y, dtype=np.int)
        assd_distance = assd(x, y, connectivity=2)
        m = dc(x, y)
        dice_list.append(m)
        hd_list.append(assd_distance)
    return dice_list, hd_list


def test(params):
    os.makedirs(params['save_folder'], exist_ok=True)
    eval_p = []

    model = CA_UNet3D(input_shape=params['input_shape'], in_channels=2, out_channels=6, init_channels=16)
    network = load_model(params['model_path'], model)
    network = network.cuda()
    network.eval()

    test_loader = make_test_loader(params['data_root'], params['index_path'], params['data_modes'],
                                   params['roi_modes'], params['input_shape'], params['crop_index'],
                                   batch_size=params['batch_size'])

    with torch.no_grad():
        for batch_x, batch_y, info in test_loader:
            name = info['case_path'][0].split('/')[-1]
            store_path = os.path.join(params['save_folder'], name,  'seg.nii.gz')
            print(store_path)
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            output = network(batch_x)
            # the shape of batch_y and output is [N, 1, H, W, D], [N, C, H, W, D]
            dice_list, hd_list = cal_dice_and_hd(output, batch_y, c=6)
            eval_p.append([name] + hd_list)
            print(hd_list)
            print(dice_list)
            save_sample(output, params['original_size'], params['crop_index'], store_path)

        with open(os.path.join(params['save_folder'], 'test_hd.csv'), 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ID', 'CN_dice', 'GP_dice', 'PUT_dice', 'SN_dice', 'RN_dice'])
            writer.writerows(eval_p)


if __name__ == '__main__':
    params = {
        'data_root':  r'/homes/ydwang/Data/stage_data_0616',
        'index_path': r'/homes/ydwang/projects/RJH_Nucleus_seg/index/test_index.csv',
        'data_modes': ['SWIM.nii', 'T1WE.nii'],
        'roi_modes': ['SWIM_ROI.nii', 'T1MAP_ROI'],
        'log_path': r'./logs',
        'crop_index': [80, 240, 100, 260],
        'input_shape': [160, 160, 64],
        'batch_size': 1,
        'original_size': [384, 384, 64],
        'save_folder': r'/homes/ydwang/projects/RJH_Nucleus_seg',
        'model_path': r'/homes/ydwang/projects/RJH_Nucleus_seg/CAunet_logs/model/Epoch240_best_model.pth'
    }
    test(params)
    