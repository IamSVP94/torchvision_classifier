import cv2
import torch
import random
import pandas as pd
from pathlib import Path
import albumentations as A
import matplotlib.pyplot as plt
import torchmetrics
from torch.utils.data import Dataset
from warmup_scheduler import GradualWarmupScheduler
from albumentations import BasicTransform, BaseCompose
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2 as ToTensor
from typing import List, Tuple, Union, Sequence


# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # del this! (need for debug)

class CustomDataset(Dataset):
    def __init__(self,
                 augmentation: Sequence[Union[BasicTransform, BaseCompose]] = None,
                 img_list: List[str] = None,
                 csv_file=None,
                 labels: List[str] = None,
                 ) -> None:
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
        min_class_counter = min(self.class_balance.values())
        self.weighted = torch.Tensor(
            [self.class_balance[i] / min_class_counter for i in range(len(self.class_balance.keys()))])

        if augmentation is None:
            augmentation = []
        augmentation.append(ToTensor(always_apply=True))
        self.augmentation = A.Compose(augmentation)

    def __len__(self) -> int:
        return len(self.imgs)

    def get_classes(self) -> int:
        return set(self.labels)

    def __getitem__(self, item: int) -> Tuple:
        filename = self.imgs[item]
        img = cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)
        # apply augmentations
        img = self.augmentation(image=img)['image'].float()
        gt = torch.Tensor([self.labels[item]]).to(torch.int64)
        return img, gt


class CommonNNModule_pl(pl.LightningModule):
    def __init__(self, model, classes, *args, loss_fn=torch.nn.CrossEntropyLoss(),
                 start_learning_rate=1e-3, checkpoint=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_learning_rate = start_learning_rate
        self.model = model
        self.classes = classes
        if checkpoint is not None and Path(checkpoint).exists():
            self.model.load_state_dict(torch.load(str(checkpoint)))
        self.loss_fn = loss_fn
        self.save_hyperparameters(ignore=['model', 'loss_fn'])

    def forward(self, img):
        pred = self.model(img)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.start_learning_rate,
            # weight_decay=1e-2,
        )

        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='min', factor=0.5,
        #     patience=10, cooldown=5,
        #     min_lr=1e-7, eps=1e-7,
        #     verbose=True,
        # )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=lr_scheduler)
        return {"optimizer": optimizer, "lr_scheduler": warmup_scheduler, "monitor": 'val_loss'}

    def train_dataloader(self):
        return self.loader['train']

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage='train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage='validation')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, stage='test')

    def predict_step(self, batch, batch_idx):
        imgs = batch  # Inference Dataset only!
        preds = self.forward(imgs)
        return preds

    def _shared_step(self, batch, stage):
        imgs, labels = batch
        gts = labels.squeeze(1).to(torch.long)
        preds = self.forward(imgs)
        loss = self.loss_fn(preds, gts)
        self.log(f'loss/{stage}', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'preds': preds}

    def save_onnx_best(self, weights, onnx_path, window_size, dynamic_hw=False):
        weights = torch.load(weights, map_location="cpu")
        self.load_state_dict(weights['state_dict'])
        self.model.to("cpu")

        input_height, input_width = window_size  # h,w format
        x = torch.randn(1, 3, input_height, input_width, requires_grad=True)

        # [b,c,h,w] - tensor.shape
        dynamic_axes = {0: "batch"}
        if dynamic_hw:
            dynamic_axes[2], dynamic_axes[3] = "height", "width"
        params = {
            'export_params': True, 'do_constant_folding': True,
            'opset_version': 11, 'input_names': ["input"], 'output_names': ["output"],
            'dynamic_axes': {'input': dynamic_axes, 'output': dynamic_axes},
        }
        torch.onnx.export(model=self.model, args=x, f=str(onnx_path), **params)


class Classifier_pl(CommonNNModule_pl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def get_random_colors(n=1):
    colors = []
    for i in range(n):
        randomcolor = (random.randint(0, 150), random.randint(50, 200), random.randint(50, 200))
        colors.append(randomcolor)
    return colors


class MetricSMPCallback(pl.Callback):
    def __init__(self, metrics, activation=None) -> None:
        self.metrics = metrics
        if activation is not None:
            self.activation = activation
        else:
            self.activation = torch.nn.Identity()

    @torch.no_grad()
    def _get_metrics(self, preds, gts, trainer):
        metric_results = {k: dict() for k in self.metrics}
        preds = torch.argmax(self.activation(preds), dim=1)

        for m_name, metric in self.metrics.items():
            metric_results[m_name] = {}
            metric = metric.to(preds.get_device())
            metric_results[m_name] = round(metric(preds, gts).item(), 4)
        return metric_results

    def plot_confusion_matrix(self, preds, gts, trainer):
        matrix = torchmetrics.ConfusionMatrix(
            task='multiclass', num_classes=len(trainer.model.classes)
        ).to(preds.get_device())(preds, gts)
        # TODO: add conf matrix in logger https://stackoverflow.com/questions/41617463/tensorflow-confusion-matrix-in-tensorboard

        # trainer.model.logger.experiment.add_image(
        #     tag=f'{stage}/{batch_idx}',
        #     img_tensor=img_for_show,
        #     global_step=trainer.current_epoch,
        #     dataformats='HWC'
        # )
        print(189, matrix)
        exit()

    @torch.no_grad()
    def _on_shared_batch_end(self, trainer, outputs, batch, batch_idx, stage) -> None:
        imgs, gts = batch
        preds = self.activation(outputs['preds'])
        metrics = self._get_metrics(preds=preds, gts=gts.squeeze(1), trainer=trainer)
        for m_name, m_val in metrics.items():
            m_title = f'{m_name}/{stage}'
            trainer.model.log(m_title, m_val, on_step=False, on_epoch=True)
        # self.plot_confusion_matrix(preds=preds, gts=gts.squeeze(1), trainer=trainer)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._on_shared_batch_end(trainer, outputs, batch, batch_idx, stage='train')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._on_shared_batch_end(trainer, outputs, batch, batch_idx, stage='validation')

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._on_shared_batch_end(trainer, outputs, batch, batch_idx, stage='test')
