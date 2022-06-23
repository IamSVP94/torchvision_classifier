from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.constants import BASE_DIR, num_workers, start_learning_rate, BATCH_SIZE, AVAIL_GPUS
from src.utils import FacesDataset, ConvNext_pl
from torchvision import models
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from datetime import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

DATASET_DIR = Path('/home/vid/hdd/projects/PycharmProjects/torchvision_classifier/temp/faces/')
imgs = list(DATASET_DIR.glob('**/*.jpg'))
labels = [0 if i.parts[-2] == 'bad' else 1 for i in imgs]

train = FacesDataset(csv_file=BASE_DIR / 'temp/train.csv', mode='train')
test = FacesDataset(csv_file=BASE_DIR / 'temp/test.csv', mode='test')

loaders = {
    'train': DataLoader(
        train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    ),
    'test': DataLoader(
        test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    ),
}

# TODO: start form checkpoint
# TODO: save checkpoint.pth
logdir = BASE_DIR / 'logs' / datetime.now().strftime("%Y%m%d-%H%M%S")
logdir.mkdir(parents=True, exist_ok=True)
logger = TensorBoardLogger(save_dir=logdir, name='Faces2206', version=0)

checkpoint_model = '/home/vid/hdd/projects/PycharmProjects/torchvision_classifier/logs/20220623-152404/Faces2206/version_0/checkpoints/last.ckpt'
model = ConvNext_pl(
    model=models.convnext_tiny(pretrained=False, num_classes=2),
    criterion=torch.nn.CrossEntropyLoss(weight=train.weighted),
    start_learning_rate=start_learning_rate,
    batch_size=BATCH_SIZE,
    loader=loaders,
    checkpoint=checkpoint_model,
)

lr_monitor = LearningRateMonitor(logging_interval='epoch')
ckp_monitor = ModelCheckpoint(monitor='val_accuracy', mode='max',
                              save_last=True, save_top_k=2,
                              save_on_train_epoch_end=True,
                              filename='{epoch:02d}-{step:02d}-{val_accuracy:.4f}',
                              )
trainer = Trainer(gpus=AVAIL_GPUS,
                  max_epochs=400,
                  logger=logger,
                  log_every_n_steps=10,
                  callbacks=[lr_monitor, ckp_monitor],
                  )

ckp_trainer = '/home/vid/hdd/projects/PycharmProjects/torchvision_classifier/logs/20220623-152927/Faces2206/version_0/checkpoints/epoch=31-step=352.ckpt'
trainer.fit(model, ckpt_path=ckp_trainer)

model.save_pth('ckp.pth')
