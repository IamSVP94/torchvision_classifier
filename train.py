from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.constants import SEED, BASE_DIR, num_workers, start_learning_rate, BATCH_SIZE, AVAIL_GPUS
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
logger = TensorBoardLogger(save_dir=logdir, name='Faces2206')
lr_monitor = LearningRateMonitor(logging_interval='epoch')

# checkpoint_model =
model = ConvNext_pl(
    model=models.convnext_tiny(pretrained=False, num_classes=2),
    # checkpoint=checkpoint_model,
    criterion=torch.nn.CrossEntropyLoss(weight=train.weighted),
    start_learning_rate=start_learning_rate,
    batch_size=BATCH_SIZE,
    loader=loaders,
)

trainer = Trainer(gpus=AVAIL_GPUS,
                  max_epochs=7,
                  logger=logger,
                  callbacks=[lr_monitor],
                  )

checkpoint_trainer = '/home/vid/hdd/projects/PycharmProjects/torchvision_classifier/logs/20220623-124116/Faces2206/version_0/checkpoints/epoch=5-step=630.ckpt'
trainer.fit(model, ckpt_path=checkpoint_trainer)
