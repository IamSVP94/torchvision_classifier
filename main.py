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

DATASET_DIR = Path('/home/vid/hdd/projects/PycharmProjects/torchvision_classifier/temp/faces/')
imgs = list(DATASET_DIR.glob('**/*.jpg'))
labels = [0 if i.parts[-2] == 'bad' else 1 for i in imgs]

X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.2, random_state=SEED, stratify=labels)
# TODO: check seed here

train = FacesDataset(img_list=X_train, labels=y_train, mode='train')
test = FacesDataset(img_list=X_test, labels=y_test, mode='test')

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

logdir = BASE_DIR / 'temp' / 'logs' / datetime.now().strftime("%Y%m%d-%H%M%S")
logdir.mkdir(parents=True, exist_ok=True)
logger = TensorBoardLogger(save_dir=logdir, name='Faces2206')
lr_monitor = LearningRateMonitor(logging_interval='step')

model = ConvNext_pl(
    model=models.convnext_tiny(pretrained=False, num_classes=2),
    criterion=torch.nn.CrossEntropyLoss,
    start_learning_rate=start_learning_rate,
    batch_size=BATCH_SIZE,
    loader=loaders,
)

trainer = Trainer(gpus=AVAIL_GPUS,
                  max_epochs=10,
                  logger=logger,
                  callbacks=[lr_monitor])

trainer.fit(model)
