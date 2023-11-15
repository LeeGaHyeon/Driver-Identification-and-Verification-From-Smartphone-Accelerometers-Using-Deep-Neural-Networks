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

def calculate_voting_3(model, dataloader):
    voting_3 = 0
    voting_3_t = 0

    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(data)
            ps = F.softmax(outputs, dim=1)

            top_p, top_class = ps.topk(1, dim=1)

        for i in range(len(labels)):
            label = labels[i]
            prediction = top_class[i].item()
            voting_3 += prediction == label
            voting_3_t += 1

    return (voting_3 / voting_3_t) * 100

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

set_seed(7777)

from sklearn.metrics import precision_recall_curve, auc

def calculate_ap(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)

def calculate_map(y_true, y_scores, num_classes):
    aps = []
    for class_id in range(num_classes):
        class_targets = [1 if label == class_id else 0 for label in y_true]
        class_scores = [score[class_id] for score in y_scores]
        aps.append(calculate_ap(class_targets, class_scores))
    return sum(aps) / len(aps)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Load your data CSV file
df = pd.read_csv('./11_longitudinal_spectrogram.csv')

# n_fold = 4  # 회차별
n_fold = 3 # 코스별

for fold in range(0, n_fold):
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold + 1}/{n_fold} ...')

    # # Split the data into training and validation sets based on the fold
    # df_train = df[df['round'] != fold + 1]
    # df_valid = df[df['round'] == fold + 1]

    df_train = df[df['course'] != fold]
    df_valid = df[df['course'] == fold]

    df_train = df_train.reset_index()
    df_valid = df_valid.reset_index()

    batch_size = 10

    train_dataset = CustomDataset(df_train, 'train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = CustomDataset(df_valid, 'valid')
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    n_epochs = 15

    best_valid_loss = np.Inf
    best_valid_accuracy = 0.0

    train_loss = torch.zeros(n_epochs)
    valid_loss = torch.zeros(n_epochs)
    train_acc = torch.zeros(n_epochs)
    valid_acc = torch.zeros(n_epochs)

    # 모델 생성
    num_classes = 11
    hidden_size = 128
    num_layers = 2

    model = ResNet50GRUClassifier(num_classes, hidden_size, num_layers).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-8,
                                                     verbose=True)

    for e in range(0, n_epochs): # train
        print(f'epoch: {e}')

        train_preds = []
        train_probs = []
        train_labels = []

        model_preds = []
        model_labels = []
        model_probs = []

        model.to(device)
        model.train()

        for data, labels in tqdm(train_loader):
            labels = labels.type(torch.LongTensor)
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss[e] += loss.item()

            ps = F.softmax(outputs, dim=1)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.reshape(top_class.shape)
            train_acc[e] += torch.mean(equals.type(torch.float)).detach().cpu()

        train_loss[e] /= len(train_loader)
        train_acc[e] /= len(train_loader)

        with torch.no_grad(): # validation
            model.eval()

            val_loss = []
            mse_loss = []
            weight_loss = []

            logit_list = []
            target_list = []

            # class 맞춘 개수
            class_correct = torch.zeros(11)
            # 전체 data를 class 별 갯수로 정리
            class_total = torch.zeros(11)

            for data, labels in tqdm(valid_loader):
                labels = labels.type(torch.LongTensor)
                data, labels = data.to(device), labels.to(device)

                logits = model(data)
                loss = criterion(logits, labels)

                valid_loss[e] += loss.item()
                val_loss.append(loss.detach().cpu().numpy())

                ps = F.softmax(logits, dim=1)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.reshape(top_class.shape)
                valid_acc[e] += torch.mean(equals.type(torch.float)).detach().cpu()

                ps = ps.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += equals[i].item()
                    class_total[label] += 1

                    logit_list.append(ps[i])
                    target_list.append(labels[i])


            voting_1 = 0
            voting_1_t = 0
            voting_3 = 0
            voting_3_t = 0
            voting_10 = 0
            voting_10_t = 0
            vote_all = 0
            vote_all_t = 0

            voting_1_top3 = 0
            voting_1_top3_t = 0
            voting_3_top3 = 0
            voting_3_top3_t = 0
            voting_10_top3 = 0
            voting_10_top3_t = 0
            voting_all_top3 = 0
            voting_all_top3_t = 0

            voting_1_TP_total = 0
            voting_1_FP_total = 0
            voting_1_FN_total = 0
            voting_3_TP_total = 0
            voting_3_FP_total = 0
            voting_3_FN_total = 0
            voting_10_TP_total = 0
            voting_10_FP_total = 0
            voting_10_FN_total = 0
            voting_all_TP_total = 0
            voting_all_FP_total = 0
            voting_all_FN_total = 0

            y_true_1, y_scores_1 = [], []
            y_true_3, y_scores_3 = [], []
            y_true_10, y_scores_10 = [], []
            y_true_all, y_scores_all = [], []

            # 사람 위치 찾기
            for i in range(len(ps[0])):
                people_index = list(filter(lambda x: target_list[x] == i, range(len(target_list))))

                if len(people_index) == 0:
                    continue  # 해당 레이블과 일치하는 데이터가 없는 경우 다음 반복으로 건너뜁니다.

                softmax = sum([logit_list[x] for x in people_index]) / len(people_index)
                vote_all += np.argmax(softmax) == i
                vote_all_t += 1

                top_3_indices = np.argsort(softmax)[-3:]
                voting_all_top3 += i in top_3_indices
                voting_all_top3_t += 1

                pred_label = np.argmax(softmax)



                # F1 점수를 위한 카운팅
                if pred_label == i:
                    voting_all_TP_total += 1
                else:
                    voting_all_FP_total += 1
                    voting_all_FN_total += 1

                y_scores_all.append(softmax[i])
                y_true_all.append(pred_label == i)

                for x in range(0, len(people_index), 1):
                    softmax_1 = sum([logit_list[j] for j in people_index[x:x + 1]])
                    voting_1 += np.argmax(softmax_1) == i
                    voting_1_t += 1

                    top_3_indices = np.argsort(softmax_1)[-3:]
                    voting_1_top3 += i in top_3_indices
                    voting_1_top3_t += 1

                    pred_label = np.argmax(softmax_1)

                    # F1 점수를 위한 카운팅
                    if pred_label == i:
                        voting_1_TP_total += 1
                    else:
                        voting_1_FP_total += 1
                        voting_1_FN_total += 1

                    y_scores_1.append(softmax_1[i])
                    y_true_1.append(pred_label == i)

                for x in range(0, len(people_index), 3):
                    softmax_3 = sum([logit_list[j] for j in people_index[x:x + 3]])
                    voting_3 += np.argmax(softmax_3) == i
                    voting_3_t += 1

                    top_3_indices = np.argsort(softmax_3)[-3:]
                    voting_3_top3 += i in top_3_indices
                    voting_3_top3_t += 1

                    pred_label = np.argmax(softmax_3)

                    # F1 점수를 위한 카운팅
                    if pred_label == i:
                        voting_3_TP_total += 1
                    else:
                        voting_3_FP_total += 1
                        voting_3_FN_total += 1

                    y_scores_3.append(softmax_3[i])
                    y_true_3.append(pred_label == i)

                for x in range(0, len(people_index), 10):
                    softmax_10 = sum([logit_list[j] for j in people_index[x:x + 10]])
                    voting_10 += np.argmax(softmax_10) == i
                    voting_10_t += 1

                    top_3_indices = np.argsort(softmax_10)[-3:]
                    voting_10_top3 += i in top_3_indices
                    voting_10_top3_t += 1

                    pred_label = np.argmax(softmax_10)

                    # F1 점수를 위한 카운팅
                    if pred_label == i:
                        voting_10_TP_total += 1
                    else:
                        voting_10_FP_total += 1
                        voting_10_FN_total += 1

                    y_scores_10.append(softmax_10[i])
                    y_true_10.append(pred_label == i)

        valid_loss[e] /= len(valid_loader)
        valid_acc[e] /= len(valid_loader)

        print('Training Loss: {:.6f} \tValidation Loss: {:.6f}'.format(train_loss[e], valid_loss[e]))
        print('Train Accuracy: {:.6f} \tValid Accuracy: {:.6f}'.format(train_acc[e], valid_acc[e]))

        if valid_loss[e] <= best_valid_loss:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(best_valid_loss,
                                                                                            valid_loss[e]))
            torch.save(model.state_dict(), f'11_round.pt')
            best_valid_loss = valid_loss[e]

        model.load_state_dict(torch.load(f'11_round.pt'))

        from sklearn.metrics import average_precision_score
        import numpy as np


        # # mAP 점수 계산 함수
        # def calculate_map_score(y_true_bin, y_scores):
        #     n_classes = y_true_bin.shape[1]
        #     map_score = 0
        #     for i in range(n_classes):
        #         ap = average_precision_score(y_true_bin[:, i], np.array(y_scores)[:, i])
        #         map_score += ap
        #     return map_score / n_classes

        from sklearn.metrics import precision_recall_curve
        import numpy as np
        from sklearn.preprocessing import label_binarize


        def calculate_map_score_modified(y_true, y_scores, n_classes):
            # y_true를 이진 레이블 행렬로 변환
            y_true_bin = label_binarize(y_true, classes=range(n_classes))

            average_precisions = []

            for i in range(n_classes):
                # 정밀도와 재현율 계산
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
                # Average Precision 계산
                ap = np.trapz(recall, precision)
                average_precisions.append(ap)

            # 모든 클래스에 대한 Average Precision의 평균 반환
            return np.mean(average_precisions)


        n_classes = 11

        y_true_bin_all = np.zeros((len(y_true_all), n_classes))
        y_scores_mat_all = np.zeros((len(y_scores_all), n_classes))

        for idx, (true_label, score) in enumerate(zip(y_true_all, y_scores_all)):
            y_true_bin_all[idx, true_label] = 1
            y_scores_mat_all[idx] = score


        # Same process for y_true_1 and y_scores_1
        y_true_bin_1 = np.zeros((len(y_true_1), n_classes))
        y_scores_mat_1 = np.zeros((len(y_scores_1), n_classes))

        for idx, (true_label, score) in enumerate(zip(y_true_1, y_scores_1)):
            y_true_bin_1[idx, true_label] = 1
            y_scores_mat_1[idx] = score

        # Calculate mAP for voting_1
        map_1 = calculate_map_score_modified(y_true_bin_1, y_scores_mat_1, n_classes)

        # Same process for y_true_1 and y_scores_1
        y_true_bin_3 = np.zeros((len(y_true_3), n_classes))
        y_scores_mat_3 = np.zeros((len(y_scores_3), n_classes))

        for idx, (true_label, score) in enumerate(zip(y_true_3, y_scores_3)):
            y_true_bin_3[idx, true_label] = 1
            y_scores_mat_3[idx] = score

        # Calculate mAP for voting_1
        map_3 = calculate_map_score_modified(y_true_bin_3, y_scores_mat_3, n_classes)

        # Same process for y_true_1 and y_scores_1
        y_true_bin_10 = np.zeros((len(y_true_10), n_classes))
        y_scores_mat_10 = np.zeros((len(y_scores_10), n_classes))

        for idx, (true_label, score) in enumerate(zip(y_true_10, y_scores_10)):
            y_true_bin_10[idx, true_label] = 1
            y_scores_mat_10[idx] = score

        # Calculate mAP for voting_1
        map_10 = calculate_map_score_modified(y_true_bin_10, y_scores_mat_10, n_classes)

        # Same process for y_true_1 and y_scores_1
        y_true_bin_all = np.zeros((len(y_true_all), n_classes))
        y_scores_mat_all = np.zeros((len(y_scores_all), n_classes))

        for idx, (true_label, score) in enumerate(zip(y_true_all, y_scores_all)):
            y_true_bin_all[idx, true_label] = 1
            y_scores_mat_all[idx] = score

        # Calculate mAP for all votes
        map_all = calculate_map_score_modified(y_true_bin_all, y_scores_mat_all, n_classes)

        # Print out the results
        print(f'VOTING1 MAP: {map_1:.4f}')
        print(f'VOTING3 MAP: {map_3:.4f}')
        print(f'VOTING10 MAP: {map_10:.4f}')
        print(f'All Votes MAP: {map_all:.4f}')

        # After training for each fold, you can calculate the voting_3 and voting_10 for that fold:
        voting_1_accuracy = (voting_1/voting_1_t)
        voting_3_accuracy = (voting_3 / voting_3_t)
        voting_10_accuracy = (voting_10 / voting_10_t)
        voting_all_accuracy = (vote_all / vote_all_t)

        print(f"voting_1_ ACC: {voting_1_accuracy}")
        print(f"voting_3_ ACC: {voting_3_accuracy}")
        print(f"voting_10_ ACC: {voting_10_accuracy}")
        print(f"voting_all_ ACC: {voting_all_accuracy}")

        # 전체 F1 점수 계산
        voting_1_precision_total = voting_1_TP_total / (voting_1_TP_total + voting_1_FP_total) if (voting_1_TP_total + voting_1_FP_total) > 0 else 0
        voting_1_recall_total = voting_1_TP_total / (voting_1_TP_total + voting_1_FN_total) if (voting_1_TP_total + voting_1_FN_total) > 0 else 0
        voting_1_f1_total = 2 * (voting_1_precision_total * voting_1_recall_total) / (voting_1_precision_total + voting_1_recall_total) if (voting_1_precision_total + voting_1_recall_total) > 0 else 0

        voting_3_precision_total = voting_3_TP_total / (voting_3_TP_total + voting_3_FP_total) if (voting_3_TP_total + voting_3_FP_total) > 0 else 0
        voting_3_recall_total = voting_3_TP_total / (voting_3_TP_total + voting_3_FN_total) if (voting_3_TP_total + voting_3_FN_total) > 0 else 0
        voting_3_f1_total = 2 * (voting_3_precision_total * voting_3_recall_total) / (voting_3_precision_total + voting_3_recall_total) if (voting_3_precision_total + voting_3_recall_total) > 0 else 0

        voting_10_precision_total = voting_10_TP_total / (voting_10_TP_total + voting_10_FP_total) if (voting_10_TP_total + voting_10_FP_total) > 0 else 0
        voting_10_recall_total = voting_10_TP_total / (voting_10_TP_total + voting_10_FN_total) if (voting_10_TP_total + voting_10_FN_total) > 0 else 0
        voting_10_f1_total = 2 * (voting_10_precision_total * voting_10_recall_total) / (voting_10_precision_total + voting_10_recall_total) if (voting_10_precision_total + voting_10_recall_total) > 0 else 0

        voting_all_precision_total = voting_all_TP_total / (voting_all_TP_total + voting_all_FP_total) if (voting_all_TP_total + voting_all_FP_total) > 0 else 0
        voting_all_recall_total = voting_all_TP_total / (voting_all_TP_total + voting_all_FN_total) if (voting_all_TP_total + voting_all_FN_total) > 0 else 0
        voting_all_f1_total = 2 * (voting_all_precision_total * voting_all_recall_total) / (voting_all_precision_total + voting_all_recall_total) if (voting_all_precision_total + voting_all_recall_total) > 0 else 0

        mAP_1 = average_precision_score(y_true_1, y_scores_1)
        print('voting_1_ MAP : ',mAP_1)

        mAP_3 = average_precision_score(y_true_3, y_scores_3)
        print('voting_3_ MAP : ', mAP_3)

        mAP_10 = average_precision_score(y_true_10, y_scores_10)
        print('voting_10_ MAP : ', mAP_10)

        mAP_all = average_precision_score(y_true_all, y_scores_all)
        print('voting_all_ MAP : ', mAP_all)

        print(f"voting_1_ F1 Score: {voting_1_f1_total}")
        print(f"voting_3_ F1 Score: {voting_3_f1_total}")
        print(f"voting_10_ F1 Score: {voting_10_f1_total}")
        print(f"voting_all_ F1 Score: {voting_all_f1_total}")

        # print(f'Voting_1 Accuracy for fold {fold + 1}: {voting_1_accuracy}%')
        print('voting_1 top3: ', voting_1_top3/voting_1_top3_t)
        # print(f'Voting_3 Accuracy for fold {fold + 1}: {voting_3_accuracy}%')
        print('voting_3 top3: ', voting_3_top3 / voting_3_top3_t)
        # print(f'Voting_10 Accuracy for fold {fold + 1}: {voting_10_accuracy}%')
        print('voting_10 top3: ', voting_10_top3 / voting_10_top3_t)
        # print(f'Voting_all Accuracy for fold {fold + 1}: {voting_all_accuracy}%')
        print('voting_all top3: ', voting_all_top3 / voting_all_top3_t)

