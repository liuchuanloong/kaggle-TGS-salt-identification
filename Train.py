from Dataset import TGS_Dataset
from Models import UNetResNet34, UNetResNet34_SE_Hyper, UNetResNet34_SE_Hyper_v2, UNetResNet34_SE, UNetResNet34_SE_Hyper_SPP, UNetResNet50_SE, FPNetResNet34, RefineNetResNet34
from contextlib import contextmanager
import time
import os

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

##############################
TRAIN_PATH = './Data/Train'
AUX_PATH = './Data/auxiliary_data'
LOAD_PATHS = None

# LOAD_PATHS = None
DEBUG = False
##############################
LOSS = 'lovasz'
OPTIMIZER = 'SGD'
PRETRAINED = True
N_EPOCH = 150
BATCH_SIZE = 32
NET = UNetResNet34_SE_Hyper_SPP
ACTIVATION = 'relu'
###########OPTIMIZER###########
LR = 1e-2
USE_SCHEDULER = 'CosineAnneling'
MILESTONES = [20, 40, 75]
GAMMA = 0.5
PATIENCE = 10
T_MAX = 70
T_MUL = 1
LR_MIN = 0
##############################
COMMENT = 'SGDR (Tmax40, Tmul1), Lovasz, relu, pretrained'

train_dataset = TGS_Dataset(TRAIN_PATH)
# train_dataset.load_images()
loaders, ids = train_dataset.yield_dataloader(num_workers=11, batch_size=BATCH_SIZE,
                                              # auxiliary_df=TGS_Dataset.create_dataset_df(AUX_PATH)
                                              )

for i, (train_loader, val_loader) in enumerate(loaders, 1):
    with timer('Fold {}'.format(i)):
        if i < 4:
            continue
        net = NET(lr=LR, debug=DEBUG, pretrained=PRETRAINED, fold=i, activation=ACTIVATION, comment=COMMENT)
        net.define_criterion(LOSS)
        net.create_optmizer(optimizer=OPTIMIZER, use_scheduler=USE_SCHEDULER, milestones=MILESTONES,
                            gamma=GAMMA, patience=PATIENCE, T_max=T_MAX, T_mul=T_MUL, lr_min=LR_MIN)

        if LOAD_PATHS is not None:
            if LOAD_PATHS[i - 1] is not None:
                net.load_model(LOAD_PATHS[i - 1])

        net.train_network(train_loader, val_loader, n_epoch=N_EPOCH)
        net.plot_training_curve(show=True)



