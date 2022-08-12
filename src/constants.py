import torch
import random
from pathlib import Path
import pytorch_lightning as pl

# print(pl.__version__)
BASE_DIR = Path(__file__).absolute().parent.parent
SEED = 2
pl.seed_everything(SEED)  # unified seed
random.seed(SEED)  # unified seed

AVAIL_GPUS = min(1, torch.cuda.device_count())
num_workers = 15  # max 16 for pc in the office
