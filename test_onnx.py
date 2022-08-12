import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.constants import BASE_DIR
import onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score


def preprocess_input(img, mean=None, std=None, input_space="RGB", size=(112, 112)):
    max_pixel_value = 255.0
    if input_space == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resizeimg = cv2.resize(img, size)

    img = resizeimg.astype(np.float32)
    if mean is not None:
        mean = np.array(mean, dtype=np.float32)
        mean *= max_pixel_value
        img -= mean

    if std is not None:
        std = np.array(std, dtype=np.float32)
        std *= max_pixel_value

        denominator = np.reciprocal(std, dtype=np.float32)
        img *= denominator

    img = np.moveaxis(img, -1, 0)
    img = img[np.newaxis, :, :, :]
    return img


model_path = '/home/vid/hdd/projects/PycharmProjects/torchvision_classifier/src/face_selector_softmaxv2.onnx'
model = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
input_name = model.get_inputs()[0].name

df = pd.read_csv('/home/vid/Downloads/datasets/face_crop_norm_dataset/datasetv2/test.csv', names=['path', 'label'])
# df_train = pd.read_csv('/home/vid/Downloads/datasets/face_crop_norm_dataset/datasetv2/train.csv', names=['path', 'label'])
# df = df_test.append(df_train, ignore_index=True)

GTs, PREDs = [], []
for idx, (path, GT) in tqdm(df.iterrows(), total=len(df), colour='green', leave=False):
    img = cv2.imread(path)  # Read image
    new_img = preprocess_input(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    PRED = model.run(None, {input_name: new_img})[0]
    GTs.append(GT)
    PREDs.append(np.argmax(PRED))

print(f'"accuracy" = {accuracy_score(GTs, PREDs)}')
print(f'"f1" = {f1_score(GTs, PREDs, average="micro")}')
