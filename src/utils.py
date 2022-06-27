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

        self.class_balance = dict()  # class balance dict
        for i in self.labels:
            self.class_balance[i] = self.class_balance[i] + 1 if self.class_balance.get(i) else 1
        min_clsas_counter = min(self.class_balance.values())
        self.weighted = torch.Tensor(
            [self.class_balance[i] / min_clsas_counter for i in range(len(self.class_balance.keys()))])
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
                A.HorizontalFlip(p=0.5),
                A.PixelDropout(p=0.2),
                A.Sharpen(p=0.2),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True),
                ToTensor(always_apply=True),
            ])
        else:
            transform = A.Compose([
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True),
                ToTensor(always_apply=True),
            ])
        GT = torch.Tensor([self.labels[item]]).to(torch.int64)
        transform_img = transform(image=resizeimg)['image'].float()
        return transform_img, GT


class ConvNext_pl(LightningModule):
    def __init__(self, model, *args, checkpoint=None,
                 start_learning_rate=1e-6, batch_size=1, criterion=nn.CrossEntropyLoss(),
                 loader=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.start_learning_rate = start_learning_rate
        self.batch_size = batch_size
        self.loader = loader
        self.model = model
        if checkpoint is not None and Path(checkpoint).exists():
            self.model.load_state_dict(torch.load(str(checkpoint)))
        self.criterion = criterion

    def forward(self, img):
        pred = self.model(img)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.start_learning_rate,
            weight_decay=1e-8,
        )

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='min', factor=0.5,
        #     patience=10, cooldown=5,
        #     min_lr=1e-7, eps=1e-7,
        #     verbose=True,
        # )

        major_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=major_scheduler)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": 'val_loss'}

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

    def save_pth(self, path, only_weight=True):
        if only_weight:
            torch.save(self.model.state_dict(), str(path))
        else:
            torch.save(self.model, str(path))
