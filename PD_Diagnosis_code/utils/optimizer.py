import torch

class Optim:
    def __init__(self, initial_lr, mode='adam'):
        self.initial_lr = initial_lr
        self.lr_scheduler_eps = 1e-3
        self.lr_scheduler_patience = 10
        self.weight_decay = 2e-5
        self.mode = mode

    def Adam(self, model):
        if self.mode == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.initial_lr)
        elif self.mode == 'adam_ReduceLROnPlateau':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, amsgrad=True)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                                              patience=self.lr_scheduler_patience,
                                              verbose=False, threshold=self.lr_scheduler_eps,
                                              threshold_mode="abs")
        else:
            optimizer =None
            print('Wrong optimizer mode!!!')
        return optimizer