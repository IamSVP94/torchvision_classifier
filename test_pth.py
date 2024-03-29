from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.constants import BASE_DIR, num_workers
from src.utils import FacesDataset
from torchvision import models
from torchmetrics.functional import accuracy, f1_score

test = FacesDataset(csv_file='/home/vid/Downloads/datasets/face_crop_norm_dataset/test.csv', mode='test')
loader = DataLoader(test, batch_size=5000, drop_last=False, shuffle=False, num_workers=num_workers)

checkpoint = '/home/vid/hdd/projects/PycharmProjects/torchvision_classifier/ckp_face_selector.pth'
model = models.convnext_tiny(pretrained=False, num_classes=2)

weights = torch.load(checkpoint, map_location="cpu")
model.load_state_dict(weights.state_dict())
# model.load_state_dict(weights)

metrics = {'accuracy': accuracy, 'f1': f1_score}

print(test.class_balance)
metrics_val = dict()
model.eval()
with torch.no_grad():
    for imgs, GTs in tqdm(loader, total=len(loader)):
        PREDs = model(imgs)
        for mname, metric in metrics.items():
            val = metric(PREDs, GTs, num_classes=2)
            if metrics_val.get(mname):
                metrics_val[mname].append(val)
            else:
                metrics_val[mname] = [val]

for k, v in metrics_val.items():  # show metric mean val
    print(f'"{k}" = {torch.mean(torch.Tensor(v))}')
