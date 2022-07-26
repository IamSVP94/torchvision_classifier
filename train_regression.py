from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from src.constants import BASE_DIR, num_workers, start_learning_rate, BATCH_SIZE, AVAIL_GPUS
from src.utils import FacesDataset, ConvNext_pl, RegressionDataset, RegressionNet_pl, VladRegressionNet
from torchvision import models
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from datetime import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

DATASET_DIR = Path(
    # '/home/vid/hdd/projects/PycharmProjects/segmentation_models.pytorch-0.1.3/temp/dataset_v9/img (copy)/')
    '/home/vid/hdd/projects/PycharmProjects/segmentation_models.pytorch-0.1.3/temp/dataset_v9/img/')

imgs = list()
for format in ['jpg', 'png', 'jpeg']:
    imgs.extend(list(DATASET_DIR.glob(f'*.{format}')))

# df_path = '/home/vid/hdd/projects/PycharmProjects/segmentation_models.pytorch-0.1.3/temp/dataset_v9/best_thresh_for_FPN_inceptionv4_best_model_maxiou_46.pth.csv'
df_path = '/home/vid/hdd/projects/PycharmProjects/segmentation_models.pytorch-0.1.3/temp/dataset_v9/2_best_thresh_for_FPN_inceptionv4_best_model_maxiou_46.pth.csv'
df = pd.read_csv(df_path, index_col=0)
GT = list()
for img_path in imgs:
    zero = df.loc[[img_path.name], '0']
    one = df.loc[[img_path.name], '1']
    GT.append([zero.item(), one.item()])

X_train, X_test, y_train, y_test = train_test_split(imgs, GT, test_size=0.33, random_state=2)

train = RegressionDataset(img_list=X_train, gt=y_train, mode='train')
test = RegressionDataset(img_list=X_test, gt=y_test, mode='test')

loaders = {
    'train': DataLoader(
        train, shuffle=True,
        batch_size=BATCH_SIZE, num_workers=num_workers,
        drop_last=False,
    ),
    'test': DataLoader(
        test, shuffle=False,
        batch_size=1, num_workers=1,
        drop_last=False,
    ),
}

# logdir = BASE_DIR / 'logs' / 'regression' / datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = BASE_DIR / 'logs' / 'regression'
logdir.mkdir(parents=True, exist_ok=True)
logger = TensorBoardLogger(save_dir=logdir, name='SmokeRegression_RegNet', version=1)

architecture = nn.Sequential(models.regnet_y_400mf(num_classes=2), nn.Sigmoid())
# checkpoint_model = '/home/vid/hdd/projects/PycharmProjects/torchvision_classifier/ckp87.pth'


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pred, gt):
        loss = torch.sqrt(self.criterion(pred, gt))
        return loss


model = RegressionNet_pl(
    model=architecture,
    criterion=RMSELoss(),
    start_learning_rate=start_learning_rate,
    batch_size=BATCH_SIZE,
    loader=loaders,
    # checkpoint=checkpoint_model,
)

lr_monitor = LearningRateMonitor(logging_interval='epoch')
ckp_val_loss = ModelCheckpoint(monitor='val_loss', mode='min',
                               save_last=True, save_top_k=2,
                               save_on_train_epoch_end=True,
                               filename='{epoch:02d}-{step:02d}-{val_loss:.4f}-{val_cos:.4f}',
                               )
ckp_cos_metric = ModelCheckpoint(monitor='val_cos', mode='max',
                                 save_last=True, save_top_k=2,
                                 save_on_train_epoch_end=True,
                                 filename='{epoch:02d}-{step:02d}-{val_cos:.4f}-{val_loss:.4f}'
                                 )

trainer = Trainer(gpus=AVAIL_GPUS,
                  max_epochs=200,
                  logger=logger,
                  log_every_n_steps=50,
                  callbacks=[lr_monitor, ckp_val_loss, ckp_cos_metric],
                  )

ckp_trainer = '/home/vid/hdd/projects/PycharmProjects/torchvision_classifier/logs/regression/SmokeRegression_RegNet/version_0/checkpoints/epoch=01-step=74-val_cos=0.1508-val_loss=0.3934.ckpt'
trainer.fit(model, ckpt_path=ckp_trainer)
# trainer.fit(model)

model.save_pth('ckp_RegNet2.pth', only_weight=False)
