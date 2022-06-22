from pathlib import Path

import torch

BASE_DIR = Path(__file__).absolute().parent.parent
SEED = 2

AVAIL_GPUS = min(1, torch.cuda.device_count())
num_workers = 1  # 15
start_learning_rate = 1e-6
BATCH_SIZE = 3 if AVAIL_GPUS else 1
