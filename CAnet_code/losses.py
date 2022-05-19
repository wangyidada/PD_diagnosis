from torch.nn import functional as F
import torch.nn as nn
import torch


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        """
        	input tesor of shape = (N, 1, H, W)
        	target tensor of shape = (N, 1, H, W)
        """
        N = target.size(0)
        smooth = 1e-2

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + 2*smooth)
        loss = 1 - loss.sum() / N
        return loss


class MulticlassDiceLoss2D(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss2D, self).__init__()

    def forward(self, input, target, weights=None):
        """
        	input tesor of shape = (N, C, H, W)
        	target tensor of shape = (N, 1, H, W)
        """
        # 先将 target 进行 one-hot 处理，转换为 (N, C, H, W)
        C = input.shape[1]
        target = torch.squeeze(target)
        traget_to_one_hot = nn.functional.one_hot(target.long(), num_classes=C)
        traget_to_one_hot = traget_to_one_hot.permute(0, 3, 1, 2)

        assert input.shape == traget_to_one_hot.shape, "predict & target shape do not match"

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0
        logits = nn.functional.softmax(input, dim=1)

        for i in range(C):
            diceLoss = dice(logits[:, i, ...], traget_to_one_hot[:, i, ...])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss/C


class MulticlassDiceLoss3D(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss3D, self).__init__()

    def forward(self, input, target, c, weights=None):
        """
        	input tesor of shape = (N, C, H, W, D)
        	target tensor of shape = (N, 1, H, W, D)
        """
        # 先将 target 进行 one-hot 处理，转换为 (N, C, H, W, D)

        target = torch.squeeze(target, dim=1)
        target_to_one_hot = nn.functional.one_hot(target.long(), num_classes=c)
        target_to_one_hot = target_to_one_hot.permute(0, 4, 1, 2, 3)
        assert input.shape == target_to_one_hot.shape, "predict & target shape do not match"

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0
        dice_list = []
        logits = nn.functional.softmax(input, dim=1)

        for i in range(1, c):
            diceLoss = dice(logits[:, i, ...], target_to_one_hot[:, i, ...])
            dice_list.append(diceLoss)
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss
        return dice_list, totalLoss/c


class MulticlassEntropyLoss2D(nn.Module):
    def __init__(self):
        super(MulticlassEntropyLoss2D, self).__init__()

    def forward(self, input, target):
        """
        	input tesor of shape = (N, C, H, W)
        	target tensor of shape = (N, 1, H, W)
        """
        C = input.size(1)
        input = input.permute(0, 2, 3, 1)
        input_flat = input.view(-1, C)
        target_flat = target.view(-1)
        loss = nn.CrossEntropyLoss()(input_flat, target_flat.long())
        return loss


class MulticlassEntropyLoss3D(nn.Module):
    def __init__(self):
        super(MulticlassEntropyLoss3D, self).__init__()

    def forward(self, input, target, c):
        """
        	input tesor of shape = (N, C, H, W, D)
        	target tensor of shape = (N, 1, H, W, D)
        """

        input = input.permute(0, 2, 3, 4, 1)
        input_flat = input.reshape(-1, c)
        target_flat = target.reshape(-1)
        loss = nn.CrossEntropyLoss()(input_flat, target_flat.long())
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)

    def forward(self, preds, labels):
        if self.ignore_index is not None:
            mask = labels != self.ignore
            labels = labels[mask]
            preds = preds[mask]

        logpt = -self.bce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss


class MulitclassFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss

def dice_loss(input, target):
    """soft dice loss"""
    eps = 1e-7
    iflat = input.reshape(-1)
    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()

    return 1 - 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)


def vae_loss(recon_x, x, mu, logvar):
    loss_dict = {}
    loss_dict['KLD'] = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss_dict['recon_loss'] = F.mse_loss(recon_x, x, reduction='mean')

    return loss_dict

def unet_vae_loss(batch_pred, batch_x, batch_y, vout, mu, logvar, LOSS_WEIGHT=0.1):
    loss_dict = {}
    loss_dict['wt_loss'] = dice_loss(batch_pred[:, 0], batch_y[:, 0])  # whole tumor
    loss_dict['tc_loss'] = dice_loss(batch_pred[:, 1], batch_y[:, 1])  # tumore core
    loss_dict['et_loss'] = dice_loss(batch_pred[:, 2], batch_y[:, 2])  # enhance tumor
    loss_dict.update(vae_loss(vout, batch_x, mu, logvar))
    weight = LOSS_WEIGHT
    loss_dict['loss'] = loss_dict['wt_loss'] + loss_dict['tc_loss'] + loss_dict['et_loss'] + \
                         weight * loss_dict['recon_loss'] + weight * loss_dict['KLD']

    return loss_dict


def total_loss(pred, target, c, weights=[0.3, 0.7]):
    """
    	a list of input tesor of shape = (N, C, H, W)
    	target tensor of shape = (N, 1, H, W)
    """
    loss_dict = {}
    dice_loss_list, dice_loss_mean = MulticlassDiceLoss3D()(pred, target, c)
    cross_entropy_loss = MulticlassEntropyLoss3D()(pred, target, c)
    total_loss = dice_loss_mean*weights[0] + cross_entropy_loss*weights[1]

    loss_dict['CN_dice'] = dice_loss_list[0]
    loss_dict['GP_dice'] = dice_loss_list[1]
    loss_dict['PUT_dice'] = dice_loss_list[2]
    loss_dict['RN_dice'] = dice_loss_list[3]
    loss_dict['SN_dice'] = dice_loss_list[4]

    loss_dict['dice_loss'] = dice_loss_mean
    loss_dict['ce_loss'] = cross_entropy_loss
    loss_dict['total_loss'] = total_loss
    return loss_dict

