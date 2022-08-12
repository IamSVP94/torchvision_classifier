'''
make 2 *.txt files with parts of dataset
'''
import random
import shutil
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

from src.constants import SEED

DATASET_DIR = Path('/home/vid/Downloads/datasets/face_crop_norm_dataset/datasetv3/')
VALID_PERCENT = 20
TEST_PERCENT = 10

imgs = list(DATASET_DIR.glob('**/*.jpg'))
random.seed(2)
random.shuffle(imgs)
labels = [i.parts[-2] for i in imgs]
le = preprocessing.LabelEncoder()
labels = le.fit_transform(labels)
le.class_info = {i: le.inverse_transform([i])[0] for i in set(labels)}
print(le.class_info)

valid_count = int(len(imgs) / 100 * VALID_PERCENT)
test_count = int(len(imgs) / 100 * TEST_PERCENT)
train_count = len(imgs) - (valid_count + test_count)

train_part = range(0, train_count)
test_part = range(train_count, train_count + test_count)
val_part = range(train_count + test_count, len(imgs))

df_train = pd.DataFrame(columns=['path', 'label'])
df_test = pd.DataFrame(columns=['path', 'label'])
df_val = pd.DataFrame(columns=['path', 'label'])
for idx, (img_path, label) in tqdm(enumerate(zip(imgs, labels)), total=len(imgs)):
    mode = "train" if idx in train_part else "test" if idx in test_part else "val"
    row = pd.Series({'path': img_path, 'label': label})
    if mode == 'train':
        df_train = df_train.append(row, ignore_index=True)
    elif mode == 'val':
        df_val = df_val.append(row, ignore_index=True)
    elif mode == 'test':
        df_test = df_test.append(row, ignore_index=True)
else:
    df_train.to_csv(DATASET_DIR / 'train.csv', header=False, index=False)
    df_val.to_csv(DATASET_DIR / 'val.csv', header=False, index=False)
    df_test.to_csv(DATASET_DIR / 'test.csv', header=False, index=False)
