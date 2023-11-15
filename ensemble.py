'''
1. 모델 및 데이터 로딩:

저장된 모델을 불러옵니다. 이 경우, 11_course_0.pt 파일입니다.
course==0인 데이터를 validation dataset으로 로딩합니다.

2. 예측 수행:

불러온 모델을 이용하여 validation dataset에 대한 예측을 수행합니다.
각 데이터 포인트에 대해 로짓 값과 실제 레이블을 저장합니다.

3. CSV 파일로 저장:

예측 결과를 포함하는 DataFrame을 생성합니다. 이 DataFrame에는 두 개의 열이 있어야 합니다: 하나는 실제 레이블을 위한 것이고, 다른 하나는 로짓 값을 위한 것입니다.
생성된 DataFrame을 CSV 파일로 저장합니다.
'''

import pandas as pd
import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import os
import cv2
import glob
import timm
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class CustomDataset(Dataset):
    def __init__(self, csv, mode):
        self.csv = csv
        self.mode = mode

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, idx):
        file = self.csv.loc[idx, 'file']
        label = torch.as_tensor(self.csv.loc[idx, 'label'], dtype=torch.long)
        file = np.load(file)

        return torch.Tensor(file).permute(2, 0, 1), label


class ResNet50GRUClassifier(nn.Module):
    def __init__(self, num_classes, hidden_size, num_layers):
        super(ResNet50GRUClassifier, self).__init__()

        self.resnet = timm.create_model('resnet50', pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(2048, hidden_size, num_layers, batch_first=True, dropout=0.5)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        features = self.resnet(x)
        features = features.view(features.size(0), -1)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(features.unsqueeze(1), h0)

        output = self.fc(out[:, -1, :])
        return output

# 모델 생성
num_classes = 11
hidden_size = 128
num_layers = 2

# 모델 불러오기
model = ResNet50GRUClassifier(num_classes, hidden_size, num_layers)
model.load_state_dict(torch.load('./model/11_course_2.pt'))
model.to(device)
model.eval()

batch_size = 10
# Validation 데이터 로드
df = pd.read_csv('./1116_longitudinal_spectrogram.csv')
df_valid = df[df['course'] == 2]
df_valid = df_valid.reset_index(drop=True)

valid_dataset = CustomDataset(df_valid, 'valid')
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# 예측 및 softmax 확률 저장
labels = []
probabilities = []

with torch.no_grad():
    for data, label in valid_loader:
        data = data.to(device)
        output = model(data)
        softmax_probs = F.softmax(output, dim=1).cpu().numpy()
        probabilities.extend(softmax_probs)
        labels.extend(label.numpy())

# 각 레이블별로 확률의 평균을 계산
unique_labels = np.unique(labels)
average_probabilities = {label: np.mean([prob for l, prob in zip(labels, probabilities) if l == label], axis=0) for label in unique_labels}


# 평균 확률을 DataFrame으로 변환
avg_probs_df = pd.DataFrame(average_probabilities).transpose()
avg_probs_df.columns = [f'{i}' for i in range(num_classes)]
avg_probs_df['val_C'] = avg_probs_df.index

# 'val_4' 열을 맨 앞으로 이동
column_order = ['val_C'] + [col for col in avg_probs_df if col != 'val_C']
avg_probs_df = avg_probs_df[column_order]

# DataFrame 생성 및 CSV로 저장
avg_probs_df.to_csv('./prob/course_C_avg_prob.csv', index=False, float_format='%.3f')

