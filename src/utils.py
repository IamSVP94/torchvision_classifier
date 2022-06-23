import random
from pathlib import Path
from typing import List, Tuple
import cv2
import pandas as pd
import torch
from torch import nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor
from warmup_scheduler import GradualWarmupScheduler
from torchmetrics.functional import accuracy
from src.constants import SEED
from sklearn import preprocessing

# print(pl.__version__)
pl.seed_everything(SEED)  # unified seed
random.seed(SEED)  # unified seed


class FacesDataset(Dataset):
    def __init__(self,
                 img_list: List[str] = None,
                 csv_file=None,
                 labels: List[str] = None,
                 input_size: Tuple[int] = (112, 112),
                 mode: str = 'test') -> None:
        assert ((csv_file is not None) or (img_list is not None)), 'You should set csv_file or img_list'

        if img_list:
            self.imgs = img_list
            self.labels = labels
        elif csv_file:
            df = pd.read_csv(csv_file, names=['path', 'label'])
            self.imgs = list(map(Path, df['path']))
            self.labels = df['label'].to_list()
        self.mode = mode
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, item: int) -> Tuple:
        filename = self.imgs[item]
        img = cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)
        resizeimg = cv2.resize(img, self.input_size)
        if self.mode == 'train':
            transform = A.Compose([
                A.HorizontalFlip(p=0.3),
                ToTensor(),
            ])
        else:
            transform = A.Compose([
                A.HorizontalFlip(p=0.3),
                ToTensor(),
            ])
        GT = torch.Tensor([self.labels[item]]).to(torch.int64)
        transform_img = transform(image=resizeimg)['image'].float()
        return transform_img, GT


class ConvNext_pl(LightningModule):
    def __init__(self,
                 model,
                 *args,
                 start_learning_rate=1e-6,
                 batch_size=1,
                 loader=None,
                 criterion=nn.CrossEntropyLoss,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.start_learning_rate = start_learning_rate
        self.batch_size = batch_size
        self.loader = loader
        self.criterion = criterion()
        self.model = model

    def forward(self, img):
        out = self.model(img)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.start_learning_rate,
            # weight_decay=1e-2,
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, cooldown=3, eps=1e-8, verbose=True)

        # TODO: ReduceLROnPlateau here
        # scheduler_steplr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=True)
        scheduler_steplr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_steplr)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.loader['train']

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        gts = labels.squeeze(1)
        preds = self(imgs)
        loss = self.criterion(preds, gts)
        self.log("train_loss", loss, prog_bar=True, batch_size=self.batch_size)
        return loss

    def val_dataloader(self):
        return self.loader['test']

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        gts = labels.squeeze(1)
        preds = self(imgs)
        loss = self.criterion(preds, gts)
        self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size)
        acc = accuracy(preds, gts)
        self.log("val_accuracy", acc, prog_bar=True, batch_size=self.batch_size)
        return loss
