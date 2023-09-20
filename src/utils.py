import io
import cv2
import torch
import random
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import albumentations as A
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from typing import List, Tuple, Union, Sequence
from warmup_scheduler import GradualWarmupScheduler
from albumentations import BasicTransform, BaseCompose
from albumentations.pytorch import ToTensorV2 as ToTensor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
        # self.save_hyperparameters(ignore=['model', 'loss_fn'])
        self.save_hyperparameters()

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
        self.steps_preds = dict()
        self.steps_gts = dict()

    def _shared__epoch_start(self, stage, *args, **kwargs) -> None:
        self.steps_preds[stage] = []
        self.steps_gts[stage] = []

    def _shared_step_minor(self, batch, batch_idx, stage):
        _, labels = batch
        gts = labels.squeeze(1).tolist()
        self.steps_gts[stage].extend(gts)

        output = self._shared_step(batch, stage=stage)
        preds = torch.argmax(output['preds'], dim=1).tolist()
        self.steps_preds[stage].extend(preds)
        return output

    def _shared_epoch_end(self, stage, *args, **kwargs) -> None:
        del self.steps_preds[stage]
        del self.steps_gts[stage]

    def on_train_epoch_start(self, *args, **kwargs) -> None:
        stage = 'train'
        self._shared__epoch_start(stage, *args, **kwargs)

    def on_validation_epoch_start(self, *args, **kwargs) -> None:
        stage = 'validation'
        self._shared__epoch_start(stage, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        stage = 'train'
        return self._shared_step_minor(batch, batch_idx, stage)

    def validation_step(self, batch, batch_idx):
        stage = 'validation'
        return self._shared_step_minor(batch, batch_idx, stage)

    def on_train_epoch_end(self, *args, **kwargs) -> None:
        stage = 'train'
        self._shared_epoch_end(stage, *args, **kwargs)

    def on_validation_epoch_end(self, *args, **kwargs) -> None:
        stage = 'validation'
        self._shared_epoch_end(stage, *args, **kwargs)


def get_random_colors(n=1):
    colors = []
    for i in range(n):
        randomcolor = (random.randint(0, 150), random.randint(50, 200), random.randint(50, 200))
        colors.append(randomcolor)
    return colors


class MetricSMPCallback(pl.Callback):
    def __init__(self, metrics, activation=None,
                 log_img: bool = False, save_img: bool = False,
                 n_img_check_per_epoch_validation: int = 0,
                 n_img_check_per_epoch_train: int = 0,
                 ) -> None:
        self.metrics = metrics
        if activation is not None:
            self.activation = activation
        else:
            self.activation = torch.nn.Identity()
        self.log_img = log_img
        self.save_img = save_img
        self.n_img_check_per_epoch_validation = n_img_check_per_epoch_validation
        self.n_img_check_per_epoch_train = n_img_check_per_epoch_train

    @torch.no_grad()
    def _get_metrics(self, preds, gts, trainer):
        metric_results = {k: dict() for k in self.metrics}
        preds = torch.argmax(self.activation(preds), dim=1)

        for m_name, metric in self.metrics.items():
            metric_results[m_name] = {}
            metric = metric.to(preds.get_device())
            metric_results[m_name] = round(metric(preds, gts).item(), 4)
        return metric_results

    @staticmethod
    def save_ax_nosave(ax, **kwargs):
        ax.axis("off")
        ax.figure.canvas.draw()
        trans = ax.figure.dpi_scale_trans.inverted()
        bbox = ax.bbox.transformed(trans)
        buff = io.BytesIO()
        plt.savefig(buff, format="png", dpi=ax.figure.dpi, bbox_inches=bbox, **kwargs)
        ax.axis("off")
        buff.seek(0)
        im = plt.imread(buff)
        return im[:, :, -2::-1] * 255

    def plot_confusion_matrix(self, preds, gts, labels=None, return_mode='img'):
        fig, ax = plt.subplots()

        matrix = confusion_matrix(gts, preds, labels=labels)

        # Set names to show in boxes
        classes = [f'gt={i} pred={j}' for i in labels for j in labels]
        # Set values format
        values = ["{0:0.0f}".format(x) for x in matrix.flatten()]
        # Find percentages and set format
        # TODO: change bug in percentages
        # percentages = ["{0:.1%}".format(x) for x in matrix.flatten() / np.sum(matrix)]

        # Combine classes, values and percentages to show
        # combined = [f"{c}\n{v} ({p})" for c, v, p in zip(classes, values, percentages)]
        combined = [f"{c}\n{v}" for c, v in zip(classes, values)]

        sns.heatmap(
            annot=np.array(combined).reshape(len(labels), len(labels)),
            data=matrix, fmt='', cmap='viridis', ax=ax,
        )
        if return_mode == 'plt':
            return fig
        elif return_mode == 'img':
            return self.save_ax_nosave(ax)

    @torch.no_grad()
    def _on_shared_batch_end(self, trainer, outputs, batch, batch_idx, stage) -> None:
        imgs, gts = batch
        preds = self.activation(outputs['preds'])
        metrics = self._get_metrics(preds=preds, gts=gts.squeeze(1), trainer=trainer)
        for m_name, m_val in metrics.items():
            m_title = f'{m_name}/{stage}'
            trainer.model.log(m_title, m_val, on_step=False, on_epoch=True)
        train_condition = stage in ['train'] and batch_idx in self.batches_check_train
        val_condition = stage in ['validation'] and batch_idx in self.batches_check_validation
        if train_condition or val_condition:
            img_for_show = self.draw_tensor(imgs[0], gts[0], preds[0])
            if self.log_img:
                trainer.model.logger.experiment.add_image(
                    tag=f'{stage}/{batch_idx}',
                    img_tensor=img_for_show,
                    global_step=trainer.current_epoch,
                    dataformats='HWC'
                )
            if self.save_img:
                log_path = Path(trainer.model.logger.experiment.get_logdir()) / 'imgs' / stage
                save_path = log_path / f'epoch={trainer.current_epoch}_batch_idx={batch_idx}.jpg'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path), cv2.cvtColor(img_for_show, cv2.COLOR_BGR2RGB))

    @torch.no_grad()
    def draw_tensor(self, img_t, gt, pred, to_numpy=True):  # need for speed
        # [-1:1]->[0:255]
        img_t = torch.permute((img_t * 0.5 + 0.5) * 255, (1, 2, 0))
        if to_numpy:
            img_t = img_t.cpu().detach().numpy().astype(np.uint8)
            title = f'g={gt.item()},p={torch.argmax(pred).item()}'
            img_t = self._cv2_add_title(img_t, title)
        return img_t

    @staticmethod
    def _cv2_add_title(img, title,
                       font=cv2.FONT_HERSHEY_COMPLEX,
                       font_scale=0.75, thickness=1,
                       where='top', color=(0, 0, 0),
                       ):
        lines_w = lines_h = 0
        lines = []
        for i, line in enumerate(title.split('\n')):
            lines.append(line)
            (text_w, text_h), _ = cv2.getTextSize(title, font, font_scale, thickness)
            lines_w += text_w
            lines_h += 2 * text_h

        if where == 'top':
            text_pos_x, text_pos_y = 5, int(lines_h / len(lines))
            top = lines_h + text_h
            bottom = left = right = 0
        if where == 'bottom':
            pass
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (255, 255, 255))

        for line in lines:
            cv2.putText(img, line, (text_pos_x, text_pos_y), font, font_scale, color, thickness)
            if where == 'top':
                text_pos_y += int(lines_h / len(lines))
        return img

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._on_shared_batch_end(trainer, outputs, batch, batch_idx, stage='train')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._on_shared_batch_end(trainer, outputs, batch, batch_idx, stage='validation')

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._on_shared_batch_end(trainer, outputs, batch, batch_idx, stage='test')

    def _shared_epoch_end(self, trainer, pl_module, stage) -> None:
        preds = pl_module.steps_preds[stage]
        gts = pl_module.steps_gts[stage]

        conf_matrix = self.plot_confusion_matrix(
            preds=preds,
            gts=gts,
            labels=sorted(list(trainer.model.classes)),
            return_mode='plt',
        )
        # plt_show_img(conf_matrix, 'cm')  # if return_mode='img'
        trainer.model.logger.experiment.add_figure(
            tag=f'{stage}/conf_matrix',
            figure=conf_matrix,
            global_step=trainer.current_epoch,
        )

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        stage = 'train'
        self._shared_epoch_end(trainer, pl_module, stage)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        stage = 'validation'
        self._shared_epoch_end(trainer, pl_module, stage)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if trainer.current_epoch == 0:  # if 0 epoch not need to save
            self.batches_check_train = []
        else:
            self.batches_check_train = random.sample(
                range(0, trainer.val_check_batch),
                min(self.n_img_check_per_epoch_train, trainer.val_check_batch)
            )

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self.batches_check_validation = random.sample(
            range(0, trainer.val_check_batch),
            min(self.n_img_check_per_epoch_validation, trainer.val_check_batch)
        )


def max_show_img_size_reshape(img, max_show_img_size, return_coef=False):  # h,w format
    img_c = img.copy()
    h, w = img_c.shape[:2]
    coef = 1
    if h > max_show_img_size[0] or w > max_show_img_size[1]:
        h_coef = h / max_show_img_size[0]
        w_coef = w / max_show_img_size[1]
        if h_coef < w_coef:  # save the biggest side
            new_img_width = max_show_img_size[1]
            coef = w / new_img_width
            new_img_height = h / coef
        else:
            new_img_height = max_show_img_size[0]
            coef = h / new_img_height
            new_img_width = w / coef
        new_img_height, new_img_width = map(int, [new_img_height, new_img_width])
        img_c = cv2.resize(img_c.astype(np.uint8), (new_img_width, new_img_height), interpolation=cv2.INTER_LINEAR)
    if return_coef:
        return img_c, coef
    return img_c


def plt_show_img(img,
                 title: str = None,
                 coef=None,
                 mode: str = 'plt',
                 max_img_size: Tuple[str] = (900, 900),
                 save_path: Union[str, Path] = None) -> None:
    if isinstance(img, (str, Path)):
        img = cv2.imread(str(img))
    img = img.astype(np.uint8)
    img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if mode == 'plt' else img
    if coef:
        img_show = img_show * coef
    img_show = img_show.copy().astype(np.uint8)
    title = str(title) if title is not None else 'image'
    if mode == 'plt':
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(img_show)
        if title:
            ax.set_title(title)
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
        fig.show()
    elif mode == 'cv2':
        if max_img_size is not None:
            img_show = max_show_img_size_reshape(img_show, max_img_size).astype(np.uint8)
        cv2.imshow(title, img_show)
        cv2.waitKey(0)
