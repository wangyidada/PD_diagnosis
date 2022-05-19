import os
import torch
from Dataset import make_data_loaders
from torch.utils.tensorboard import SummaryWriter
from utils.early_stopping import EarlyStopping
from models.AG_SE_ResNeXt50 import AG_SE_ResNeXt50_model
from models.SE_ResNeXt import seresnext50
from models.ResNet50 import resnet50
from models.vgg16 import vgg16
import tqdm
from utils.optimizer import Optim
from sklearn.metrics import auc, roc_curve
from utils.losses import FocalLoss


def train_val(model, loaders, optimizer, criterion, early_stopping, log_path, batch_size,  n_epochs=10):
    writer = SummaryWriter(os.path.join(log_path, 'log_dir'))
    for epoch in range(n_epochs):
        train_loss, eval_loss = 0, 0
        train_acc, eval_acc = 0, 0
        loss_dict, acc_dict, auc_dict = {}, {}, {}
        train_pred_list, train_label_list = [], []
        eval_pred_list, eval_label_list = [], []

        for phase in ['train', 'eval']:
            loader = loaders[phase]
            for (batch_img, batch_label, batch_mask, _) in tqdm.tqdm(loader):
                batch_img, batch_mask, batch_label = batch_img.cuda(), batch_mask.cuda(), batch_label.cuda()
                with torch.set_grad_enabled(phase == 'train'):
                    class_prob = model(batch_img)
                    prediction = torch.max(class_prob, dim=1)[1]
                    acc = (prediction == batch_label).sum().item()
                    class_logits = torch.sigmoid(class_prob)
                    prediction_list = list(class_logits.cpu().detach().numpy()[:, 1])
                    batch_label_list = list(batch_label.cpu().detach().numpy())
                    loss = criterion(class_prob[:, 1], batch_label)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_acc = train_acc + acc
                    train_loss = train_loss + loss.data.item()
                    train_pred_list += prediction_list
                    train_label_list += batch_label_list

                else:
                    eval_acc = eval_acc + acc
                    eval_loss = eval_loss + loss.data.item()
                    eval_pred_list += prediction_list
                    eval_label_list += batch_label_list

            if phase == 'train':
                train_loss_mean = train_loss / len(loader)
                train_acc_mean = train_acc / (len(loader)*batch_size)
                loss_dict[phase] = train_loss / len(loader)
                acc_dict[phase] = train_acc_mean
                fpr, tpr, t = roc_curve(train_label_list, train_pred_list)
                train_auc = auc(fpr, tpr)
                auc_dict[phase] = train_auc

                print('\nTrain: Epoch is : {}, loss is :{}, accuracy is {}, auc is {} '
                      .format(epoch, train_loss_mean, train_acc_mean, train_auc))

            else:
                eval_loss_mean = eval_loss / len(loader)
                eval_acc_mean = eval_acc / (len(loader)*batch_size)
                loss_dict[phase] = eval_loss_mean
                acc_dict[phase] = eval_acc_mean
                fpr, tpr, t = roc_curve(eval_pred_list, eval_label_list)
                eval_auc = auc(fpr, tpr)
                auc_dict[phase] = eval_auc
                print('\nEval: Epoch is : {}, loss is :{}, accuracy is {}, auc is {} '
                      .format(epoch, eval_loss_mean, eval_acc_mean, eval_auc))

                state = {}
                state['model'] = model.state_dict()
                state['optimizer'] = optimizer.state_dict()
                model_path = os.path.join(log_path, 'model')
                os.makedirs(model_path, exist_ok=True)
                if (epoch + 1) % 10 == 0:
                    file_name = os.path.join(model_path, 'epoch' + str(epoch + 1) + '_model.pth')
                    torch.save(state, file_name)
                early_stopping(eval_loss, state, epoch, model_path)

            writer.add_scalars('accuracy',  acc_dict, epoch)
            writer.add_scalars('auc',  auc_dict, epoch)
            writer.add_scalars('loss_total',  loss_dict, epoch)

        if early_stopping.early_stop:
            print('Early stopping')
            break

    writer.close()
    return model


def main(params):
    model = AG_SE_ResNeXt50_model(num_classes=2)
    # model = seresnext50(num_classes=2)
    # model = vgg16(num_classes=2)
    if torch.cuda.is_available():
        model = model.cuda()
        print('use_gpu')
    else:
        model = model.to('cpu')
        print('use_cpu')

    loaders = make_data_loaders(params["data_root"], params["train_csv_file"], params["val_csv_file"],
                                params["data_modes"], params["roi_modes"], params["crop_index"], params["input_shape"], batch_size=params["batch_size"])

    optimizer = Optim(initial_lr=params["lr"], mode='adam').Adam(model)
    criterion = FocalLoss().cuda()
    early_stopping = EarlyStopping(patience=params["patience"], verbose=True)
    train_val(model, loaders, optimizer, criterion, early_stopping,  params["log_path"], params["batch_size"], n_epochs=params['epoches'])


if __name__ == '__main__':
    params = {
        "data_root": r'/homes/ydwang/Data/data',
        "train_csv_file":  r'/homes/ydwang/projects/RJ_PD_dignosis/index/5_fold_up/fold_0/train_index.csv',
        "val_csv_file": r'/homes/ydwang/projects/RJ_PD_dignosis/index/5_fold_up/fold_0/val_index.csv',
        "log_path": '/homes/ydwang/projects/RJ_PD_dignosis/model/fold_0',
        "data_modes": ['QSM.nii'],
        "roi_modes": ['seg.nii.gz'],
        "crop_index": [80, 240, 100, 260],
        "input_shape": [160, 160, 64],
        "batch_size": 16,
        "n_classes": 2,
        "patience": 20,
        "epoches": 500,
        'lr': 1e-3
    }
    main(params)



