from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.constants import BASE_DIR, num_workers, AVAIL_GPUS
from src.utils import FacesDataset, Classifier_pl
from torchvision import models
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from datetime import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

DATASET_DIR = Path('/home/vid/Downloads/datasets/face_crop_norm_dataset/datasetv2/')

train = FacesDataset(csv_file='/home/vid/Downloads/datasets/face_crop_norm_dataset/datasetv2/train.csv', mode='train')
test = FacesDataset(csv_file='/home/vid/Downloads/datasets/face_crop_norm_dataset/datasetv2/test.csv', mode='test')

start_learning_rate = 1e-5
BATCH_SIZE = 270 if AVAIL_GPUS else 1

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
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    ),
}

# TODO: save checkpoint.pth while train
# logdir = BASE_DIR / 'logs/classificator_norm' / datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = BASE_DIR / 'logs/classificator_norm'
logdir.mkdir(parents=True, exist_ok=True)
version_title = f'efficientnet_b2_epochs=241_weightdecay=None_lr={start_learning_rate}'
logger = TensorBoardLogger(save_dir=logdir, name='faces_selector1108', version=version_title)

print(train.class_balance)  # class imbalance
print(train.weighted)  # class imbalance

# checkpoint_model = '/home/vid/hdd/projects/PycharmProjects/torchvision_classifier/ckp_face_selectorv2_1008_model.pth'
model = Classifier_pl(
    # model=models.convnext_tiny(pretrained=False, num_classes=2),
    model=models.efficientnet_b2(pretrained=False, num_classes=2),
    criterion=torch.nn.CrossEntropyLoss(weight=train.weighted),
    start_learning_rate=start_learning_rate,
    batch_size=BATCH_SIZE,
    loader=loaders,
    # checkpoint=checkpoint_model,
)

lr_monitor = LearningRateMonitor(logging_interval='epoch')
ckp_acc = ModelCheckpoint(monitor='val_accuracy', mode='max',
                          save_last=True, save_top_k=2,
                          save_on_train_epoch_end=True,
                          filename='{epoch:02d}-{step:02d}-{val_accuracy:.4f}-{val_loss:.4f}',
                          )
ckp_val_loss = ModelCheckpoint(monitor='val_loss', mode='min',
                               save_last=True, save_top_k=2,
                               save_on_train_epoch_end=True,
                               filename='{epoch:02d}-{step:02d}-{val_loss:.4f}-{val_accuracy:.4f}',
                               )

trainer = Trainer(gpus=AVAIL_GPUS,
                  max_epochs=241,
                  logger=logger,
                  log_every_n_steps=10,
                  callbacks=[lr_monitor, ckp_acc, ckp_val_loss],
                  )

# ckp_trainer = '/home/vid/hdd/projects/PycharmProjects/torchvision_classifier/logs/classificator_norm/faces_selector1108/convnext_tiny_500_weightdecay=0.01_lr=1e-05/checkpoints/last.ckpt'
# trainer.fit(model, ckpt_path=ckp_trainer)
trainer.fit(model)

model.save_pth(f'{version_title}.pth')
