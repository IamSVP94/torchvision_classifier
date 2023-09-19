import torch
import torchmetrics
from pathlib import Path
from clearml import Task
import albumentations as A
from torchvision import models
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from src.constants import BASE_DIR, num_workers, AVAIL_GPUS
from src.utils import Classifier_pl, CustomDataset, MetricSMPCallback
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# PARAMS
DATASET_DIR = Path('/home/vid/hdd/datasets/FACES/face_selector_dataset/')
input_size = (112, 112)  # h,w
EPOCHS = 1000
start_learning_rate = 1e-6
BATCH_SIZE = 290 if AVAIL_GPUS else 1

EXPERIMENT_NAME = 'face_selector_efficientnet_b2'  # use this model from torchvision
MODEL_VERSION = 'R2.1'
VERSION_TITLE = f'{EXPERIMENT_NAME}_{MODEL_VERSION}_epochs={EPOCHS}'

logdir = BASE_DIR / 'logs/face_selector'
logdir.mkdir(parents=True, exist_ok=True)

weights = None

# CLEARML CONFIG
additional_tags = ['efficientnet_b2', ]
task = Task.init(
    project_name='FACEID/face_selector',
    task_name=EXPERIMENT_NAME, tags=['classification', MODEL_VERSION] + additional_tags,
)

# AUGMENTATIONS
train_transforms = [
    A.HorizontalFlip(p=0.5),
    A.PixelDropout(p=0.05),
    A.Sharpen(p=0.2),
    A.HueSaturationValue(p=0.3),
    A.CLAHE(p=0.3),
    # A.RandomContrast(p=0.3), # A.RandomBrightness(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.RandomGamma(p=0.3),
]
val_transforms = []

common_transforms = [
    A.Resize(height=input_size[0], width=input_size[1], always_apply=True),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True),
]

# DATASETS
train = CustomDataset(csv_file='/home/vid/hdd/datasets/FACES/face_selector_dataset/train.csv',
                      augmentation=train_transforms + common_transforms)
val = CustomDataset(csv_file='/home/vid/hdd/datasets/FACES/face_selector_dataset/test.csv',
                    augmentation=val_transforms + common_transforms)

train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, drop_last=False)
val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False)

# METRICS
metrics_callback = MetricSMPCallback(metrics={
    'accuracy': torchmetrics.classification.BinaryAccuracy(),
    'F1score': torchmetrics.classification.BinaryF1Score(),
    'auroc': torchmetrics.classification.BinaryAUROC(),
}, )

# LOGGER
tb_logger = TensorBoardLogger(save_dir=logdir, name='faces_selector', version=VERSION_TITLE)
lr_monitor = LearningRateMonitor(logging_interval='epoch')
best_metric_saver = ModelCheckpoint(
    mode='max', save_top_k=1, save_last=True, auto_insert_metric_name=False,
    monitor='accuracy/validation', filename='epoch={epoch:02d}-accuracy_val={accuracy/validation:.4f}',
)

# MODEL
# TODO: change for hugging face models using
classes = train.get_classes()
model = Classifier_pl(
    # model=models.convnext_tiny(weights=None, num_classes=len(classes)),
    model=models.efficientnet_b2(weights=None, num_classes=len(classes)),
    loss_fn=torch.nn.CrossEntropyLoss(weight=train.weighted),
    start_learning_rate=start_learning_rate, classes=classes,
)

# TRAIN
trainer = Trainer(
    max_epochs=EPOCHS, num_sanity_val_steps=0, devices=-1,
    accelerator='cuda', logger=tb_logger, log_every_n_steps=10,
    callbacks=[lr_monitor, metrics_callback, best_metric_saver],
)

trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=weights)

# TEST AND SAVE
# test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=15)
# test_metrics = trainer.test(model, dataloaders=test_loader, ckpt_path='best', verbose=True)  # ckpt_path='last'

logger_weights_dir = Path(best_metric_saver.best_model_path).parent.parent
ONNX_PATH = logger_weights_dir / f'{EXPERIMENT_NAME}_{MODEL_VERSION}_{input_size[1]}x{input_size[0]}.onnx'

# TODO: change to current logdir onnx path because rewriting?
model.save_onnx_best(
    weights=best_metric_saver.best_model_path,  # best_iou_saver.last_model_path
    window_size=input_size,
    onnx_path=ONNX_PATH,
)
