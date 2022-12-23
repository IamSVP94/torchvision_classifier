import json
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
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
from random import shuffle


# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # del this! (need for debug)

class AgeDataset(Dataset):
    def __init__(self,
                 img_list: List[str] = None,
                 num_classes: int = None,
                 len_limit: int = None,
                 csv_file=None,
                 labels: List[str] = None,
                 input_size: Tuple[int] = (112, 112),
                 mode: str = 'test') -> None:
        assert ((csv_file is not None) or (img_list is not None)), 'You should set csv_file or img_list'

        self.mode = mode
        self.num_classes = num_classes
        self.input_size = input_size

        if img_list:
            self.imgs = img_list
            self.labels = labels
        elif csv_file:
            df = pd.read_csv(csv_file, names=['path', 'label'])
            self.imgs = list(map(Path, df['path']))
            self.labels = [float(age) / 100 for age in df['label'].to_list()]  # for rergression
            # self.labels = [float(age) for age in df['label'].to_list()]  # for classification
        if len_limit:
            self.imgs = self.imgs[:len_limit]
            self.labels = self.labels[:len_limit]

        '''  # classes balance
        if mode == 'train':
            self.class_balance = dict()  # class balance dict
            for i in self.labels:
                self.class_balance[i] = self.class_balance[i] + 1 if self.class_balance.get(i) else 1
            min_clsas_counter = min(self.class_balance.values())

            self.weighted = torch.Tensor(
                [self.class_balance[i] / min_clsas_counter for i in range(len(self.class_balance.keys()))])
        '''

    def __len__(self) -> int:
        return len(self.imgs)

    @staticmethod
    def get_training_augmentation():
        train_transform = [
            A.HorizontalFlip(p=0.5),
            A.PixelDropout(p=0.2),
            A.Sharpen(p=0.2),
            A.HueSaturationValue(p=0.3),
            A.CLAHE(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomGamma(p=0.3),
        ]
        return train_transform

    def __getitem__(self, item: int) -> Tuple:
        filename = self.imgs[item]
        img = cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)

        # w, h, c = img.shape
        # if (w, h) != self.input_size:
        #     img = cv2.resize(img, self.input_size)

        transform = self.get_training_augmentation() if self.mode == 'train' else []  # augment for train
        transform.extend([
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True), ToTensor(always_apply=True),
        ])  # common preprocessing part
        transform = A.Compose(transform)

        GT = torch.Tensor([self.labels[item]]).to(torch.float)  # attention on dtype!
        transform_img = transform(image=img)['image'].float()

        return transform_img, GT


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
        self.weighted = torch.Tensor([self.class_balance[i] / min_clsas_counter for i in range(len(self.class_balance.keys()))])
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
    def get_training_augmentation_old():
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

    @staticmethod
    def get_training_augmentation():
        train_transform = [
            A.HorizontalFlip(p=0.5),
            A.PixelDropout(p=0.2),
            A.Sharpen(p=0.2),
            A.HueSaturationValue(p=0.3),
            A.CLAHE(p=0.3),
            # A.RandomContrast(p=0.3),
            # A.RandomBrightness(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomGamma(p=0.3),
        ]
        return train_transform

    def __getitem__(self, item: int) -> Tuple:
        img = self.imgs[item]
        gt = self.gt[item]

        transform = self.get_training_augmentation() if self.mode == 'train' else []

        transform.extend([
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True),
            ToTensor(always_apply=True)
        ])
        transform = A.Compose(transform)

        transform_img = transform(image=img)['image'].float()

        return transform_img, gt


class TxtDataset(Dataset):
    def __init__(self, dir):
        self.dir = dir
        txts = list(Path(dir).glob('**/*.txt'))
        shuffle(txts)
        X, y = [], []
        for txt_path in txts:
            cl = 0 if txt_path.parent.name == 'sit' else 1
            with open(txt_path, 'r') as rf:
                txt_points = json.loads(rf.readlines()[0])

                # txt_points = np.array(txt_points)
                # txt_points = txt_points.reshape(txt_points.shape[0] * txt_points.shape[1])
                txt_points = self.get_normalize_coord(txt_points)

            X.append(txt_points)
            y.append([cl])
        self.X = torch.Tensor(np.array(X))
        self.y = torch.Tensor(y)

        self.class_balance = dict()  # class balance dict
        values, counts = torch.unique(self.y, return_counts=True)
        for k, v in zip(values, counts):
            self.class_balance[int(k)] = int(v)
        min_clsas_counter = min(self.class_balance.values())
        self.weighted = torch.Tensor(
            [self.class_balance[i] / min_clsas_counter for i in range(len(self.class_balance.keys()))])

    #
    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, item: int) -> Tuple:
        # print(self.X[item])
        # print(self.y[item])
        return self.X[item], self.y[item]

    @staticmethod
    def euclidean_dist(a, b, mode='one'):
        assert mode in ['one', 'many']
        if mode == 'one':
            a = np.array([a])
            b = np.array([b])
        # This function calculates the euclidean distance between 2 point in 2-D coordinates
        # if one of two points is (0,0), dist = 0
        # a, b: input array with dimension: m, 2
        # m: number of samples
        # 2: x and y coordinate
        try:
            if (a.shape[1] == 2 and a.shape == b.shape):
                # check if element of a and b is (0,0)
                bol_a = (a[:, 0] != 0).astype(int)
                bol_b = (b[:, 0] != 0).astype(int)
                dist = np.linalg.norm(a - b, axis=1)
                return ((dist * bol_a * bol_b).reshape(a.shape[0], 1))
        except:
            print("[Error]: Check dimension of input vector")
            return 0

    @staticmethod
    def get_normalize_coord(points):
        shoulders = [int((points[5][0] + points[6][0]) / 2), int((points[5][1] + points[6][1]) / 2)]
        eyes = [int((points[1][0] + points[6][0]) / 2), int((points[2][1] + points[6][1]) / 2)]
        length_head = max(
            TxtDataset.euclidean_dist(shoulders, eyes),  # left shoulders and eyes
            TxtDataset.euclidean_dist(points[3], points[4]),  # left ear and ear
        )[0][0]
        length_torso = max(
            TxtDataset.euclidean_dist(points[5], points[11]),  # left shoulder and hip
            TxtDataset.euclidean_dist(points[6], points[12]),  # right shoulder and hip
        )[0][0]
        # print(length_torso)

        length_left_leg = TxtDataset.euclidean_dist(points[11], points[13]) + TxtDataset.euclidean_dist(points[13],
                                                                                                        points[15])
        length_right_leg = TxtDataset.euclidean_dist(points[12], points[14]) + TxtDataset.euclidean_dist(points[14],
                                                                                                         points[16])
        length_legs = max(
            length_left_leg,
            length_right_leg
        )[0][0]

        length_body = length_head + length_torso + length_legs
        # print(length_body)

        center_x = sum([point[0] / len(points) for point in points])
        center_y = sum([point[1] / len(points) for point in points])
        # print([center_x, center_y])

        norm_points = []
        for point in points:
            if point[0] != point[1] != 0:
                new_point = [(point[0] - center_x) / length_body, (point[1] - center_y) / length_body]
            else:
                new_point = [0, 0]
            norm_points.append(new_point)
        arr = np.array(norm_points)
        # return (arr - np.min(arr)) / np.ptp(arr)  # [0;1]

        normal_arr = []
        for arr_p, n_arr_p in zip(arr, (arr - np.min(arr)) / np.ptp(arr)):
            if arr_p[0] != arr_p[1] != 0:
                normal_arr.append(n_arr_p)
            else:
                normal_arr.append(arr_p)
        else:
            normal_arr = np.array(normal_arr)
        return normal_arr.reshape(normal_arr.shape[0] * normal_arr.shape[1])


class CommonNNModule_pl(LightningModule):
    def __init__(self, model, *args, checkpoint=None,
                 start_learning_rate=1e-6, batch_size=1,
                 criterion=nn.CrossEntropyLoss(), special_criterion=None,
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
        self.special_criterion = special_criterion

    def forward(self, img):
        pred = self.model(img)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.start_learning_rate,
            weight_decay=1e-2,
        )

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='min', factor=0.5,
        #     patience=10, cooldown=5,
        #     min_lr=1e-7, eps=1e-7,
        #     verbose=True,
        # )

        major_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        # scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=major_scheduler)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": 'val_loss'}
        return {"optimizer": optimizer, "lr_scheduler": major_scheduler, "monitor": 'val_loss'}

    def train_dataloader(self):
        return self.loader['train']

    def training_step(self, batch, batch_idx):
        imgs, gts = batch
        preds = self(imgs)
        loss = self.criterion(preds, gts)
        return loss

    def training_epoch_end(self, outputs):
        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("train_loss", avg_loss, prog_bar=True, batch_size=self.batch_size, on_step=False, on_epoch=True)

    def val_dataloader(self):
        return self.loader['test']

    def validation_step(self, batch, batch_idx):
        imgs, gts = batch
        preds = self(imgs)
        loss = self.criterion(preds, gts)
        return loss

    def validation_epoch_end(self, outputs):
        # calculating average loss
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log("val_loss", avg_loss, prog_bar=True, batch_size=self.batch_size, on_step=False, on_epoch=True)

    def save_pth(self, path, mode='torch', input_size=None, only_weight=True):
        assert mode in ['torch', 'onnx']
        if mode == 'torch':
            if only_weight:
                torch.save(self.model.state_dict(), str(path))
            else:
                torch.save(self.model, str(path))
        elif mode == 'onnx' and input_size:
            x = torch.randn(1, 3, input_size[0], input_size[1], requires_grad=True)
            torch.onnx.export(self.model, x, path,
                              export_params=True,
                              opset_version=11,
                              do_constant_folding=True,
                              input_names=['input'],  # the model's input names
                              output_names=['output'],
                              )


class RegressionNet_pl(CommonNNModule_pl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        imgs, gts = batch
        preds = self(imgs)
        if self.special_criterion is not None:
            loss1 = self.special_criterion(preds, gts)
            loss2 = self.criterion(preds, gts)
            loss = loss1 + loss2
        else:
            loss = self.criterion(preds, gts)
        # self.log("train_loss", loss, prog_bar=True, batch_size=self.batch_size, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, gts = batch
        preds = self(imgs)
        if self.special_criterion is not None:
            loss1 = self.criterion(preds, gts)
            loss2 = self.special_criterion(preds, gts)
            loss = loss1 + loss2
        else:
            loss = self.criterion(preds, gts)
        # self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size, on_step=False, on_epoch=True)

        mae = torch.abs(gts - preds) * 100
        self.log("val_mae", mae, prog_bar=True, batch_size=self.batch_size, on_step=False, on_epoch=True)
        # cos = CosineSimilarity(reduction='mean')(preds, gts)  # = 1 if identical
        # self.log("val_cos", 1 - cos, prog_bar=True, batch_size=self.batch_size, on_step=False, on_epoch=True)
        return loss


class Classifier_pl(CommonNNModule_pl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        gts = labels.squeeze(1).to(torch.long)
        preds = self(imgs)
        if self.special_criterion is not None:
            mean_loss, variance_loss = self.special_criterion(preds, gts)
            m = torch.nn.Softmax(dim=1)
            p = m(preds)
            softmaxloss = self.criterion(p, gts)  # должно быть после softmax?
            loss = mean_loss + variance_loss + softmaxloss
        else:
            loss = self.criterion(preds, gts)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        gts = labels.squeeze(1).to(torch.long)
        preds = self(imgs)
        if self.special_criterion is not None:
            mean_loss, variance_loss = self.special_criterion(preds, gts)
            m = torch.nn.Softmax(dim=1)
            p = m(preds)
            softmaxloss = self.criterion(p, gts)  # должно быть после softmax?
            loss = mean_loss + variance_loss + softmaxloss
        else:
            loss = self.criterion(preds, gts)

        argmaxpred = torch.argmax(preds, dim=1)
        mae = torch.abs(gts - argmaxpred).to(torch.float32)
        self.log("val_mae", mae, prog_bar=True, batch_size=self.batch_size, on_step=False, on_epoch=True)

        acc = accuracy(argmaxpred, gts)
        self.log("val_accuracy", acc, prog_bar=True, batch_size=self.batch_size, on_step=False, on_epoch=True)
        return loss


class MLP(nn.Module):
    """A multi-layer perceptron module.
    This module is a sequence of linear layers plus activation functions.
    The user can optionally add normalization and/or dropout to each of the layers.
    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        hidden_dims (Optional[List[int]]): Output dimension for each hidden layer.
        dropout (float): Probability for dropout layer.
        activation (Callable[..., nn.Module]): Which activation
            function to use. Supports module type or partial.
        normalization (Optional[Callable[..., nn.Module]]): Which
            normalization layer to use (None for no normalization).
            Supports module type or partial.
    Inputs:
        x (Tensor): Tensor containing a batch of input sequences.
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            hidden_dims: Optional[Union[int, List[int]]] = None,
            dropout: float = 0.5,
            activation: Callable[..., nn.Module] = nn.ReLU,
            normalization: Optional[Callable[..., nn.Module]] = None,
            **kwargs,
    ) -> None:
        super().__init__()

        layers = nn.ModuleList()
        # layers.append(
        #     nn.ConvTranspose2d(in_channels=2, out_channels=1,
        #                        kernel_size=1, padding=0, stride=1,
        #                        output_padding=0)
        # )

        # TODO: add ConvTranspose2d

        if hidden_dims is None:
            hidden_dims = []

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if normalization:
                layers.append(normalization(hidden_dim))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.shape)
        out = self.model(x)
        print(out)
        exit()
        return out
        # out = torch.nn.Sigmoid()(out)
        # return torch.argmax(out).unsqueeze()
        # return torch.unsqueeze(torch.argmax(out).to(int), 0)
