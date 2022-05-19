from Dataset import make_data_loaders
from models.network import CA_UNet3D
import os
import torch
from losses import total_loss
from torch.utils.tensorboard import SummaryWriter
from utils.early_stopping import EarlyStopping
import tqdm


def train_val(model, loaders, optimizer,  early_stopping, device, log_path, n_epochs=300):
    writer = SummaryWriter(os.path.join(log_path, 'log_dir'))

    for epoch in range(n_epochs):
        train_loss, eval_loss, train_dice, eval_dice = 0, 0, 0, 0
        total_loss_dict, total_dice_dict, dice_sigle = {}, {}, {}
        for phase in ['train', 'eval']:
            loader = loaders[phase]
            for (batch_x, batch_y, batch_z) in tqdm.tqdm(loader):
                batch_x, batch_y = batch_x.cuda(device[0]), batch_y.cuda(device[0])
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(batch_x)
                    loss_dict = total_loss(output, batch_y, c=6)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss_dict['total_loss'].backward()
                    optimizer.step()
                    train_loss += loss_dict['total_loss'].data.item()
                    train_dice += loss_dict['dice_loss'].data.item()

                else:
                    eval_loss += loss_dict['total_loss'].data.item()
                    eval_dice += loss_dict['dice_loss'].data.item()

            if phase == 'train':
                train_loss_mean = train_loss / len(loader)
                train_dice_mean = train_dice / len(loader)
                total_loss_dict[phase] = train_loss_mean
                total_dice_dict[phase] = train_dice_mean

                print('\nTrain: Epoch is : {}, loss is :{}, dice is {}'
                      .format(epoch, train_loss_mean, train_dice_mean))

            if phase == 'eval':
                eval_loss_mean = eval_loss / len(loader)
                eval_dice_mean = eval_dice / len(loader)
                total_loss_dict[phase] = eval_loss_mean
                total_dice_dict[phase] = eval_dice_mean
                print('\nEval: Epoch is : {}, loss is :{}, dice is {}'
                      .format(epoch, eval_loss_mean, eval_dice_mean))

                state = {}
                state['model'] = model.state_dict()
                state['optimizer'] = optimizer.state_dict()
                model_path = os.path.join(log_path, 'model')
                os.makedirs(model_path, exist_ok=True)
                if (epoch + 1) % 10 == 0:
                    file_name = os.path.join(model_path, 'epoch' + str(epoch + 1) + '_model.pth')
                    torch.save(state, file_name)
                early_stopping(eval_loss_mean, state, epoch, model_path)

        writer.add_scalars('loss', total_loss_dict, epoch)
        writer.add_scalars('dice', total_dice_dict, epoch)

        if early_stopping.early_stop:
            print('Early stopping')
            break
    writer.close()
    return model


def main(params):
    train_csv_path = os.path.join(params['index_path'], 'train_index.csv')
    val_csv_path = os.path.join(params['index_path'], 'val_index.csv')
    loaders = make_data_loaders(params['data_root'], train_csv_path, val_csv_path,
                                params['data_modes'], params['roi_modes'], params['input_shape'],
                                params['crop_index'], batch_size=params['batch_size'])

    network = CA_UNet3D(input_shape=params['input_shape'], in_channels=2, out_channels=6, init_channels=16)

    if len(params['device_ids']) > 1:
        model = torch.nn.DataParallel(network, device_ids=params['device_ids'])
        model = model.cuda(device=params['device_ids'][0])
    else:
        model = network.cuda(device=params['device_ids'][0])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    early_stopping = EarlyStopping(patience=20, verbose=True)
    train_val(model, loaders, optimizer, early_stopping, params['device_ids'], params['log_path'],  n_epochs=300)


if __name__ == "__main__":
    params = {
        'device_ids': [0, 1, 2, 3],
        'data_root': '/home/wyd/PycharmProjects/RJH/data',
        'index_path': r'/home/wyd/PycharmProjects/seg-pytorch/index',
        'data_modes': ['QSM.nii', 'T1WE.nii'],
        'roi_modes': ['QSM_ROI.nii', 'T1MAP_ROI'],
        'log_path': r'./CAunet_logs',
        'crop_index': [80, 240, 100, 260],
        'input_shape': [160, 160, 64],
        'batch_size': 4,
    }
    main(params)