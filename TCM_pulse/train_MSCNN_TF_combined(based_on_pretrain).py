import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, confusion_matrix
import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, KFold
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import datetime
import json
from torch.optim.lr_scheduler import LambdaLR
from MSCNN_TF_combined import MSCNNTran
import pickle
from datetime import datetime
from scipy.signal import butter, filtfilt
import pandas as pd
import re
from data_processor import data_preparation



class PulseDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        target = self.targets[index]
        return sample, target
    
parser = argparse.ArgumentParser(description='TCM_pulse')


# data loader
parser.add_argument('--folds', type=int, default='5', help='number of folds for cross validation')
parser.add_argument('--random_seed', type=int, default=42, help='random seed for dataset split')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file path')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# training settings
parser.add_argument('--device', type=str, default='cuda:0', help='device to use for training')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'],
                    help='Optimizer to use (default: Adam)')
parser.add_argument('--lr', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--decay_rate', type=float, default=0.98, help='decay rate of learning rate')
parser.add_argument('--min_lr_ratio', type=float, default=1e-3, 
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--warmup_epochs', type=int, default=5, 
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--stop_epochs', type=int, default=10, help='number of epochs to train')


# model define
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--channel_independence', type=int, default=1,
                    help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default='avg',
                    help='down sampling method, only support avg, max, conv')
parser.add_argument('--use_future_temporal_feature', type=int, default=0,
                    help='whether to use future_temporal_feature; True 1 False 0')
args = parser.parse_args()

def create_lr_lambda(warm_up_epochs, decay_rate, min_lr_ratio):
    def lr_lambda(epoch):
        decay_start_epoch = warm_up_epochs
        if epoch < warm_up_epochs:
            # Warm up stage
            return (epoch + 1) / warm_up_epochs
        elif epoch < decay_start_epoch:
            # after Warm up keep the learning rate
            return 1
        else:
            # Decay stage:
            decay_lr = decay_rate ** (epoch - decay_start_epoch + 1)
            return max(decay_lr, min_lr_ratio)
    return lr_lambda

def get_optimizer(optimizer_name, model_parameters, lr):
    if optimizer_name == 'SGD':
        return optim.SGD(model_parameters, lr=lr)
    elif optimizer_name == 'Adam':
        return optim.Adam(model_parameters, lr=lr)
    elif optimizer_name == 'RMSprop':
        return optim.RMSprop(model_parameters, lr=lr)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')

def matrix_compute(all_predictions, all_targets, all_logits) :

    fpr, tpr, thresholds = roc_curve(all_targets, all_logits)
    auroc = auc(fpr, tpr)
    precision, recall, thresholds = precision_recall_curve(all_targets, all_logits)
    auprc = auc(recall, precision)
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='binary')
    recall = recall_score(all_targets, all_predictions, average='binary')
    tn, fp, fn, tp = confusion_matrix(all_targets, all_predictions).ravel()
    specificity = tn / (tn + fp)
    f1 = f1_score(all_targets, all_predictions, average='binary')
    return auroc, auprc, accuracy, precision, recall, specificity, f1

def evaluation(model, data_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    all_logits = []
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, _, outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            logits = outputs.cpu().numpy()[:, 1]
            all_logits.extend(logits.tolist())
            all_predictions.extend(predicted.tolist())
            all_targets.extend(labels.tolist())
        auroc, auprc, accuracy, precision, recall, specificity, f1 = matrix_compute(all_predictions, all_targets, all_logits)
    return auroc, auprc, accuracy, precision, recall, specificity, f1, total_loss/ len(data_loader), all_logits, all_targets


def stratified_split(data_labels, test_ratio=0.2, val_ratio=0.2, random_state=42):
    # 获取0和1的索引
    data_labels = np.array(data_labels)
    indices_0 = np.where(data_labels == 0)[0]
    indices_1 = np.where(data_labels == 1)[0]

    # 定义训练集、验证集和测试集的比例
    train_ratio = 1-test_ratio - val_ratio

    # 分别对0和1的索引进行分层抽样
    def stratified_split1(indices, train_ratio, val_ratio):
        np.random.shuffle(indices)
        train_size = int(train_ratio * len(indices))
        val_size = int(val_ratio * len(indices))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        return train_indices, val_indices, test_indices

    # 对0和1的索引分别进行划分
    train_indices_0, val_indices_0, test_indices_0 = stratified_split1(indices_0, train_ratio, val_ratio)
    train_indices_1, val_indices_1, test_indices_1 = stratified_split1(indices_1, train_ratio, val_ratio)

    # 合并0和1的索引以得到最终的训练集、验证集和测试集的索引
    train_indices = np.concatenate([train_indices_0, train_indices_1])
    val_indices = np.concatenate([val_indices_0, val_indices_1])
    test_indices = np.concatenate([test_indices_0, test_indices_1])

    return train_indices, val_indices, test_indices

# 以条为单位进行训练并计算模型指标
def train_1(train_data, train_label, val_data, val_label, test_data, test_label, model, folds=args.folds, random_seed=args.random_seed, batch_size=args.batch_size, min_lr_ratio=args.min_lr_ratio,
          epochs=args.epochs, stop_epochs=args.stop_epochs, optimizer_name=args.optimizer, device=args.device, warm_up_epochs=args.warmup_epochs, lr=args.lr, decay_rate=args.decay_rate):
    
    # 将数据转换为PyTorch张量
    train_data = torch.tensor(train_data, dtype=torch.double)
    train_labels = torch.tensor(train_label, dtype=torch.long)
    val_data = torch.tensor(val_data, dtype=torch.double)
    val_labels = torch.tensor(val_label, dtype=torch.long)
    test_data = torch.tensor(test_data, dtype=torch.double)
    test_labels = torch.tensor(test_label, dtype=torch.long)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device).double()
    
    # 结果保存
    best_auroc_5_fold = []
    best_auprc_5_fold = []
    best_f1_5_fold = []
    now = datetime.now()
    formatted_date = now.strftime("%m-%d %H:%M:%S")
    # 分层抽样
    # train_index, val_index, test_index = stratified_split(label_patient, test_ratio=0.2, validation_ratio=0.2,random_state=random_seed)

    model = MSCNNTran(1, 2)
    model.to(device).double()
    # 这个函数用来生成index，使得不同病人的数据能分开,不存在数据泄露
##################################################################
    def expand_array(original_array):
        original_array = (original_array)*4
        # 指定序列和重复次数
        sequence = [0, 1, 2, 3]
        repetitions = len(original_array)
        # 使用tile函数重复序列
        new_array = np.tile(sequence, repetitions)
        # 将原始数组的值添加到新数组的每一行的开始
        new_array = np.repeat(original_array, 4) + new_array.flatten()
        return new_array

    train_dataset = PulseDataset(train_data, train_labels)
    val_dataset = PulseDataset(val_data, val_labels)
    test_dataset = PulseDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last = False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last = False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last = False)
    
    decay_rate = args.decay_rate
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr)
    # waim up
    # 创建lr_lambda函数
    lr_lambda = create_lr_lambda(warm_up_epochs, decay_rate, min_lr_ratio)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()
    
    best_loss = 100
    
    count = 0
    for epoch in range(epochs):
        print()
        print('Epoch {}/{}'.format(epoch+1, epochs))
        model.train()

        train_loss = 0
        all_predictions = []
        all_targets = []
        all_logits = []    

        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)  # move to GPU
            optimizer.zero_grad()
            head1, head2, outputs = model(inputs, pretrain=False)
            # loss = criterion(outputs, labels)
            loss = 0.5 * criterion(outputs, labels) + 0.5 * criterion(head1, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
    
            _, predicted = torch.max(outputs, 1)
            logits = outputs.detach().cpu().numpy()[:, 1]
            all_logits.extend(logits.tolist())
            all_predictions.extend(predicted.tolist())
            all_targets.extend(labels.tolist())

        # learning rate scheduling
        scheduler.step()
        # model performance on training set
        avg_train_loss = train_loss / len(train_loader)
        auroc_tra, auprc_tra, accuracy_tra, precision_tra, recall_tra, specificity_tra, f1_tra = matrix_compute(all_predictions, all_targets, all_logits)
  
        print(f'Learning Rate: {scheduler.get_last_lr()[0]}')
        print('Train:')
        print(f"AUROC:{auroc_tra:.4f}, AUPRC:{auprc_tra:.4f}, Accuracy:{accuracy_tra:.4f}, Precision: {precision_tra:.4f}, Recall: {recall_tra:.4f}, Specificity: {specificity_tra:.4f}, F1 Score: {f1_tra:.4f}, Train Loss: {avg_train_loss:.4f}")

        # model performance on val set
        auroc_val, auprc_val, accuracy_val, precision_val, recall_val, specificity_val, f1_val, avg_val_loss1, all_logits_val, all_targets_val = evaluation(model, val_loader, device)
        print('val:')
        print(f"AUROC:{auroc_val:.4f}, AUPRC:{auprc_val:.4f}, Accuracy:{accuracy_val:.4f}, Precision: {precision_val:.4f}, Recall: {recall_val:.4f}, Specificity: {specificity_val:.4f}, F1 Score: {f1_val:.4f}, val Loss: {avg_val_loss1:.4f}")
        
        # model performance on test set
        auroc_test, auprc_test, accuracy_test, precision_test, recall_test, specificity_test, f1_test, avg_test_loss1, all_logits_test, all_targets_test = evaluation(model, test_loader, device)
        print('test:')
        print(f"AUROC:{auroc_test:.4f}, AUPRC:{auprc_test:.4f}, Accuracy:{accuracy_test:.4f}, Precision: {precision_test:.4f}, Recall: {recall_test:.4f}, Specificity: {specificity_test:.4f}, F1 Score: {f1_test:.4f}, test Loss: {avg_test_loss1:.4f}")

        # save the best model
        save_path = " "
        if epoch > int(0.5*epochs):  
            if avg_val_loss1 < best_loss:
                best_loss = avg_val_loss1
                best_auroc_tra = auroc_tra
                best_auprc_tra = auprc_tra
                best_auroc_val = auroc_val
                best_auprc_val = auprc_val
                best_auroc_test = auroc_test
                best_auprc_test = auprc_test
                best_auprc = auprc_val
                best_f1 = f1_val
                best_model = model.state_dict()
                count = 0
                record_epoch = epoch
            else:
                count += 1
                if count == stop_epochs:
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    torch.save(best_model, save_path + f"/{formatted_date} best_model_on_epoch{epoch+1}.pth")
                    np.save(save_path + f'/{formatted_date} val_logits', all_logits_val)
                    np.save(save_path + f'/{formatted_date} val_labels', all_targets_val)
                    np.save(save_path + f'/{formatted_date} test_logits', all_logits_test)
                    np.save(save_path + f'/{formatted_date} test_labels', all_targets_test)
                                      
                    print(f"Best model saved at epoch {record_epoch+1} with \n train_AUROC&AUPRC:{best_auroc_tra:.4f} {best_auprc_tra:.4f} \n val_AUROC&AUPRC:{best_auroc_val:.4f} {best_auprc_val:.4f} \n test_AUROC&AUPRC:{best_auroc_test:.4f} {best_auprc_test:.4f}")
                    break


if __name__ == '__main__':

    state_dict = torch.load('best_model.pth')   # pretrained model
    model = MSCNNTran(1, 2)
    model.load_state_dict(state_dict)
    
    data_path1 = '/pulse_data/train_dict.pkl'
    data_path2 = '/pulse_data/val_dict.pkl'
    data_path3 = '/pulse_data/test_dict.pkl'    
    train_data, train_label, label_patient = data_preparation(data_path1)
    val_data, val_label, label_patient = data_preparation(data_path2)
    test_data, test_label, label_patient = data_preparation(data_path3)
    
    train_data = np.expand_dims(train_data, axis=1)
    val_data = np.expand_dims(val_data, axis=1)
    test_data = np.expand_dims(test_data, axis=1)

    train_1(train_data, train_label, val_data, val_label, test_data, test_label, model)
 
