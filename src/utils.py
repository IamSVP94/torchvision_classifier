import random
from typing import List, Tuple

import cv2
import torch
from torch import nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor
from warmup_scheduler import GradualWarmupScheduler

from src.constants import SEED

# print(pl.__version__)
pl.seed_everything(SEED)  # unified seed
random.seed(SEED)  # unified seed


class FacesDataset(Dataset):
    def __init__(self,
                 img_list: List[str],
                 labels: List[str] = None,
                 input_size: Tuple[int] = (112, 112),
                 mode: str = 'test') -> None:
        self.imgs = img_list
        self.labels = labels
        self.mode = mode
        self.input_size = input_size  # h,w
        self._iter = 0

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, item: int) -> dict:
        output = dict()
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
        output['original_img'] = img
        output['path'] = str(filename)
        output['label'] = torch.Tensor([self.labels[item]])
        output['img'] = transform(image=resizeimg)['image']
        self._iter += 1
        print(52, self.mode)  # TODO: del str
        return output


class ConvNext_pl(LightningModule):
    def __init__(self,
                 model, num_classes: int,
                 *args,
                 pretrained=False,
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
        self.model = model(pretrained=pretrained, num_classes=num_classes)

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
        imgs, labels = batch['img'], batch['label']
        out = self.model(imgs)
        preds = torch.argmax(out, dim=1)
        loss = self.criterion(preds, labels)
        self.log("train_loss", loss, prog_bar=True)  # TODO: add metric Accuracy plot
        return loss

    def val_dataloader(self):
        return self.loader['test']

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch['img'], batch['label']
        out = self.model(imgs.float())
        print('\n', 112)
        print(labels.shape)
        print(out.shape)
        loss = self.criterion(out, labels)
        self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size)  # TODO: add metric Accuracy plot
        return loss