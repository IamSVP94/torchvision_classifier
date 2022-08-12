from pathlib import Path
from typing import List, Tuple
import cv2
import pandas as pd
import torch
from torch import nn as nn
from pytorch_lightning import LightningModule
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor
from warmup_scheduler import GradualWarmupScheduler
from torchmetrics.functional import accuracy
from torchmetrics import CosineSimilarity


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
                A.HueSaturationValue(p=0.3),
                A.CLAHE(p=0.3),
                # A.RandomContrast(p=0.3),
                # A.RandomBrightness(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.RandomGamma(p=0.3),
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


class Classifier_pl(LightningModule):
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
            # weight_decay=0.01,
        )

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='min', factor=0.5,
        #     patience=10, cooldown=5,
        #     min_lr=1e-7, eps=1e-7,
        #     verbose=True,
        # )

        # major_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        major_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=major_scheduler)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": 'val_loss'}

    def train_dataloader(self):
        return self.loader['train']

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        gts = labels.squeeze(1)
        preds = self(imgs)
        loss = self.criterion(preds, gts)
        self.log("train_loss", loss, prog_bar=True, batch_size=self.batch_size, on_step=False, on_epoch=True)
        return loss

    def val_dataloader(self):
        return self.loader['test']

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        gts = labels.squeeze(1)
        preds = self(imgs)
        loss = self.criterion(preds, gts)
        self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size, on_step=False, on_epoch=True)
        acc = accuracy(preds, gts)
        self.log("val_accuracy", acc, prog_bar=True, batch_size=self.batch_size, on_step=False, on_epoch=True)
        return loss

    def save_pth(self, path, mode='torch', only_weight=True):
        assert mode in ['torch', 'onnx']
        if mode == 'torch':
            if only_weight:
                torch.save(self.model.state_dict(), str(path))
            else:
                torch.save(self.model, str(path))
        elif mode == 'onnx':
            pass
            # dummy_input = torch.randn(1, 3, 224, 224)
            # torch.onnx.export(self.model, dummy_input, "vgg.onnx")


class RegressionDataset(Dataset):
    def __init__(self,
                 img_list: List[str] = None,
                 gt: List[float] = None,
                 input_size: Tuple[int] = (1920, 1088),
                 mode: str = 'test') -> None:
        self.imgs = list()
        self.imgs_path = list()
        for img_path in img_list:  # this is wrong (only for small datasets!)
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            resizeimg = cv2.resize(img, input_size)
            self.imgs.append(resizeimg)
            self.imgs_path.append(img_path)
        self.gt = list(map(torch.Tensor, gt))
        self.mode = mode
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.imgs)

    @staticmethod
    def get_training_augmentation():
        train_transform = [

            A.HorizontalFlip(p=0.5),

            A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

            # A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
            # A.CropNonEmptyMaskIfExists(height=1088, width=1088, always_apply=True, p=1.0),
            # A.Resize(height=256, width=256, always_apply=True),

            # A.IAAAdditiveGaussianNoise(p=0.2),
            # A.IAAPerspective(p=0.5),

            # A.OneOf(
            #     [
            #         A.CLAHE(p=1),
            #         A.RandomBrightness(p=1),
            #         A.RandomGamma(p=1),
            #     ],
            #     p=0.9,
            # ),

            A.OneOf(
                [
                    A.GridDistortion(p=1),
                    A.ElasticTransform(p=1),
                ],
                p=0.5,
            ),

            A.OneOf(
                [
                    A.Sharpen(p=1),
                    A.Blur(blur_limit=3, p=1),
                    A.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.9,
            ),

            # A.OneOf(
            #     [
            #         A.RandomContrast(p=1),
            #         A.HueSaturationValue(p=1),
            #     ],
            #     p=0.9,
            # ),

            ToTensor(always_apply=True),
        ]
        return A.Compose(train_transform)

    def __getitem__(self, item: int) -> Tuple:
        img = self.imgs[item]
        gt = self.gt[item]
        if self.mode == 'train':
            transform = self.get_training_augmentation()
        else:
            transform = A.Compose([
                ToTensor(always_apply=True),
            ])
        transform_img = transform(image=img)['image'].float()
        return transform_img, gt


class VladRegressionNet(nn.Module):
    def __init__(self, class_num=2, activation=nn.ReLU()):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        # self.fc1 = nn.Linear(16 * 5 * 5, class_num * 3)
        self.fc1 = nn.Linear(2064960, class_num * 4)
        self.fc2 = nn.Linear(class_num * 4, class_num * 2)
        self.fc3 = nn.Linear(class_num * 2, class_num)
        self.activation = activation

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        x = nn.Sigmoid()(x)  # for values diapason in [0,1]
        return x


class RegressionNet_pl(LightningModule):
    def __init__(self, model, *args, checkpoint=None,
                 start_learning_rate=1e-6, batch_size=1, criterion=nn.CrossEntropyLoss(),
                 loader=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.start_learning_rate = start_learning_rate
        self.batch_size = batch_size
        self.model = model
        if checkpoint is not None and Path(checkpoint).exists():
            self.model.load_state_dict(torch.load(str(checkpoint)))
        self.criterion = criterion
        self.loader = loader

    def forward(self, img):
        pred = self.model(img)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.start_learning_rate,
            # weight_decay=1e+4,
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
        imgs, gts = batch
        preds = self(imgs)
        loss = self.criterion(preds, gts)
        self.log("train_loss", loss, prog_bar=True, batch_size=self.batch_size, on_step=False, on_epoch=True)
        return loss

    def val_dataloader(self):
        return self.loader['test']

    def validation_step(self, batch, batch_idx):
        imgs, gts = batch
        preds = self(imgs)
        loss = self.criterion(preds, gts)
        self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size, on_step=False, on_epoch=True)
        cos = CosineSimilarity(reduction='mean')(preds, gts)  # = 1 if identical
        self.log("val_cos", 1 - cos, prog_bar=True, batch_size=self.batch_size, on_step=False, on_epoch=True)
        return loss

    def save_pth(self, path, only_weight=True):
        if only_weight:
            torch.save(self.model.state_dict(), str(path))
        else:
            torch.save(self.model, str(path))
