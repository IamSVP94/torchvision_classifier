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

DATASET_DIR = Path('/home/vid/hdd/projects/PycharmProjects/torchvision_classifier/temp/faces/')
TEST_PERCENT = 20

imgs = list(DATASET_DIR.glob('**/*.jpg'))
labels = [i.parts[-2] for i in imgs]

le = preprocessing.LabelEncoder()
labels = le.fit_transform(labels)
le.class_info = {i: le.inverse_transform([i])[0] for i in set(labels)}

train_imgs, test_imgs, train_labels, test_labels = train_test_split(imgs, labels,
                                                                    test_size=TEST_PERCENT / 100,
                                                                    random_state=SEED, stratify=labels)

train = pd.DataFrame({'path': train_imgs, 'label': train_labels})
test = pd.DataFrame({'path': test_imgs, 'label': test_labels})

train.sample(frac=1).to_csv(DATASET_DIR.parent / 'train.csv', header=False, index=False)
test.sample(frac=1).to_csv(DATASET_DIR.parent / 'test.csv', header=False, index=False)
