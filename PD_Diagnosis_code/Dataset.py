import os
from torch.utils.data import Dataset, DataLoader
from utils.my_utils import normalize, clahe_img, standardization
from Get_data import get_data
import volumentations as volumentations
import pandas as pd
import numpy as np
import torch


class Datasets3D(Dataset):
    def __init__(self, data_folder, csv_file, data_modes, roi_modes, input_shape, crop_index,
                 is_training=True, is_normalize=False, is_CLAHE=False, input_feature=False):
        self.data_folder = data_folder
        self.data_modes = data_modes
        self.csv_file = csv_file
        self.case_list = self.get_case_list(self.csv_file)
        self.roi_modes = roi_modes
        self.resize_shape = input_shape
        self.is_training = is_training
        self.is_CLAHE = is_CLAHE
        self.is_normlize = is_normalize
        self.input_feature = input_feature
        self.crop_index = crop_index

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, index):
        case = self.case_list[index]
        path = os.path.join(self.data_folder, case)
        data_list, seg_list = get_data(path, self.data_modes, self.roi_modes, self.crop_index)
        data_volumes = np.stack(data_list, axis=-1).astype('float32')
        seg_volumes = np.stack(seg_list, axis=-1).astype("float32")
        aug_data_volumes, aug_seg_volumes = self.aug_sample(data_volumes, seg_volumes, is_training=self.is_training)
        normlized_data_volumes = self.normlize_data(aug_data_volumes)
        # data_input, mask_input: [c, h, w, d]
        data_input = np.transpose(normlized_data_volumes, [3, 0, 1, 2]).astype('float32')
        mask_input = np.squeeze(aug_seg_volumes)
        data_input = np.expand_dims(data_input, axis=1)
        # mask_input = np.transpose(aug_seg_volumes, [3, 0, 1, 2]).astype('float32')

        if self.input_feature:
            label, feature = self.get_feature(case, self.csv_file)
            label_input = np.asarray(label, dtype=np.float32)
            feature_input = np.asarray(feature, dtype=np.float32)
            mask_input = torch.nn.functional.one_hot(torch.tensor(mask_input.copy(), dtype=torch.int64))
            mask_input = mask_input.permute(3, 0, 1, 2)

            return (torch.tensor(data_input.copy(), dtype=torch.float),
                    torch.tensor(feature_input.copy(), dtype=torch.float),
                    torch.tensor(label_input.copy(), dtype=torch.float),
                    torch.tensor(mask_input.copy(), dtype=torch.float),
                    case)
        else:
            label = self.get_feature(case, self.csv_file)
            label_input = np.asarray(label, dtype=np.float32)

            mask_input = torch.nn.functional.one_hot(torch.tensor(mask_input.copy(), dtype=torch.int64))
            mask_input = mask_input.permute(3, 0, 1, 2)
            mask_input = mask_input.float()

            return (torch.tensor(data_input.copy(), dtype=torch.float),
                    torch.tensor(label_input.copy(), dtype=torch.float),
                    mask_input, case)

    def aug_sample(self, image, mask, is_training=True):
        """
        vol: [H, W, D(, C)]

        x, y, z <--> H, W, D

        you should give (H, W, D) form shape.

        skimage interpolation notations:

        order = 0: Nearest-Neighbor
        order = 1: Bi-Linear (default)
        order = 2: Bi-Quadratic
        order = 3: Bi-Cubic
        order = 4: Bi-Quartic
        order = 5: Bi-Quintic

        Interpolation behaves strangely when input of type int.
        ** Be sure to change volume and mask data type to float !!! **
        I change resize in functionals.py!!!
        """
        image = np.float32(image)
        mask = np.float32(mask)
        if is_training:
            train_tranform = volumentations.Compose([
                volumentations.Resize(self.resize_shape, interpolation=3, always_apply=True, p=1.0),
                # volumentations.RandomCrop(self.resize_shape, p=1),
                volumentations.Rotate((0, 0), (0, 0), (-5, 5), interpolation=3, p=0.5),
                volumentations.Flip(0, p=0.5),
                volumentations.Flip(1, p=0.5),
                volumentations.Flip(2, p=0.5),
                volumentations.RandomScale(scale_limit=[0.9, 1.1], interpolation=1, p=0.5),
                volumentations.PadIfNeeded(self.resize_shape, border_mode='constant', always_apply=True, p=1.0),
                volumentations.ElasticTransform((0, 0.1), interpolation=3, p=0.2),
                volumentations.RandomGamma((0.9, 1.1), p=0.2),
            ], p=1.0)
        else:
            train_tranform = volumentations.Compose([
                volumentations.Resize(self.resize_shape, interpolation=1, always_apply=True)], p=1.0)
        data = {'image': image, 'mask': mask}
        transformed_data = train_tranform(**data)
        aug_image = transformed_data['image']
        aug_mask = transformed_data['mask']
        return aug_image, aug_mask

    def get_feature(self, case, csv_file):
        df = pd.read_csv(csv_file)
        feature = df[df['ID'] == case].values[0][1:]
        assert len(feature) != 0, 'Wrong Clinical file!!!'

        if len(feature) == 1:
            label = feature[0]
            return label
        else:
            label = feature[0]
            feature = feature[1:]
            return label, feature

    def get_case_list(self, csv_file):
        df = pd.read_csv(csv_file)
        id = df['ID'].values
        case_list = list(id)
        return case_list

    def normlize_data(self, data):
        num = np.shape(data)[-1]
        if self.is_CLAHE:
            n_data_list = [clahe_img(data[..., i]) for i in range(num)]
            n_data = np.stack(n_data_list, axis=3)
        elif self.is_normlize:
            n_data_list = [normalize(data[..., i]) for i in range(num)]
            n_data = np.stack(n_data_list, axis=3)
        else:
            n_data_list = [standardization(data[..., i]) for i in range(num)]
            n_data = np.stack(n_data_list, axis=3)
        return n_data


def make_data_loaders(data_root, train_csv_file, val_csv_file, data_modes, roi_modes, crop_index, input_shape, batch_size=8):
    train_ds = Datasets3D(data_root, train_csv_file, data_modes, roi_modes, input_shape, crop_index, is_CLAHE=True)
    val_ds = Datasets3D(data_root, val_csv_file, data_modes, roi_modes, input_shape, crop_index, is_training=False, is_CLAHE=True)
    loaders = {}
    loaders['train'] = DataLoader(train_ds, batch_size=batch_size,
                              num_workers=0, shuffle=True, drop_last=True)
    loaders['eval'] = DataLoader(val_ds, batch_size=batch_size,
                             num_workers=0, shuffle=True, drop_last=True)
    return loaders


def make_test_loader(data_root, test_csv_file, data_modes, roi_modes, crop_index, input_shape, batch_size=1):
    test_ds = Datasets3D(data_root, test_csv_file, data_modes, roi_modes, input_shape, crop_index, is_training=False, is_CLAHE=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                                 num_workers=0, shuffle=False)
    return test_loader


if __name__ == "__main__":
    data_root = r'/homes/ydwang/Data/stage_data_0616'
    data_modes = ['QSM.nii', 'T1MAP.nii']
    roi_modes = ['seg.nii.gz']
    crop_index = [80, 240, 100, 260]
    input_shape = [160, 160, 64]

    index_path= r'/homes/ydwang/projects/RJ_PD_dignosis/index/5_fold_up/fold_0'

    train_csv_file = os.path.join(index_path, 'train_index.csv')
    val_csv_file = os.path.join(index_path, 'val_index.csv')
    test_csv_file = os.path.join(index_path, 'test_index.csv')

    loader = make_test_loader(data_root,  val_csv_file, data_modes,
                                roi_modes, crop_index, input_shape, batch_size=1)

    for x, y, z, name in loader:
        images_data = x.numpy()
        seg_data = z.numpy()
        label = y.numpy()
        print(np.unique(seg_data))
        print(images_data.shape, seg_data.shape)
        print(label)

        import matplotlib.pyplot as plt
        plt.imshow(images_data[0, 0, 0, :, :, 25], cmap='gray')
        plt.show()
        plt.imshow(images_data[0, 1, 0, :, :, 25], cmap='gray')
        plt.show()
        plt.imshow(seg_data[0,  0, :, :, 25], cmap='gray')
        plt.show()




