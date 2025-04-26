import os
from tqdm import tqdm
from itertools import chain
import pandas as pd
import random
import torch
import numpy as np
from  matplotlib import pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# TensorFlow 2.x中设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from datetime import datetime, timedelta


folder_path = 'D:\\fire\data\\2500'
time = []
# 遍历文件夹中的所有文件
def get_data(folder_path):
    global time
    train = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            # 训练集                 
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path,dtype=float)
            df = df.set_index('Time').sort_index()
            if(filename != '340.csv'):
                train.append(df)
            else:
                test = df
    time = test.index
    return train, test;
    # 获取训练数据和测试数据
train_data,test_data = get_data(folder_path)

# 数据集处理，获取对应的数据以及标签
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
       
def my_data(split,data):

    scaler = MinMaxScaler()
    seq = []
    if split == 'train':    
        for i in range(len(data)):
            x = data[i]
            # 归一化
            normalized_data = scaler.fit_transform(x)
            for j in range(0,18):
                for i in range(len(normalized_data) - 110):
                    train_seq,train_label = [],[]
                    for k in range(i,i+100):
                        train_seq.append([normalized_data[k,j],normalized_data[k,j+18]])
                    train_label.append([normalized_data[i + 110,j], normalized_data[i + 110,j+18]])
                    train_seq = torch.FloatTensor(train_seq).reshape(-1,2)
                    train_label = torch.FloatTensor(train_label).reshape(-1,2)
                    seq.append((train_seq, train_label))

        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=200, shuffle=False, num_workers=0, drop_last=True)
        return seq

    else:
        # split
        scaler = MinMaxScaler()

        x = data
        # 归一化
        normalized_data = scaler.fit_transform(x)
        for i in range(len(normalized_data) - 110):
            test_seq = []
            test_label = []
            for k in range(i,i+100):
                test_seq.append([normalized_data[k,0],normalized_data[k,18]])
            test_label.append([normalized_data[i + 110,0], normalized_data[i + 110,18]])
            test_seq = torch.FloatTensor(test_seq).reshape(-1,2)
            test_label = torch.FloatTensor(test_label).reshape(-1,2)
            seq.append((test_seq, test_label))
        
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

        return seq


def my_data_adddate(split,data):

    scaler = MinMaxScaler()
    seq = []
    if split == 'train':    
        for i in range(len(data)):
            start_datetime = datetime(2024, 1, 1, 0, 0, 0)

            # 创建一个列表来存储日期时间
            datetime_list = [start_datetime + timedelta(minutes=i) for i in range(2500)]
            x = data[i]
            # 归一化
            normalized_data = scaler.fit_transform(x)
            for j in range(0,18):
                for i in range(len(normalized_data) - 110):
                    train_seq,train_label = [],[]
                    for k in range(i,i+100):
                        train_seq.append([normalized_data[k,j],normalized_data[k,j+18]])
                    train_label.append([normalized_data[i + 110,j], normalized_data[i + 110,j+18]])
                    train_seq = torch.FloatTensor(train_seq).reshape(-1,2)
                    train_label = torch.FloatTensor(train_label).reshape(-1,2)
                    seq.append((train_seq, train_label))

        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=200, shuffle=False, num_workers=0, drop_last=True)
        return seq

    else:
        # split
        scaler = MinMaxScaler()
        
        start_datetime = datetime(2024, 1, 1, 0, 0, 0)

            # 创建一个列表来存储日期时间
        datetime_list = [start_datetime + timedelta(minutes=i) for i in range(2500)]

        x = data
        # 归一化
        normalized_data = scaler.fit_transform(x)
        for i in range(len(normalized_data) - 110):
            test_seq = []
            test_label = []
            for k in range(i,i+100):
                test_seq.append([normalized_data[k,0],normalized_data[k,18]])
            test_label.append([normalized_data[i + 110,0], normalized_data[i + 110,18]])
            test_seq = torch.FloatTensor(test_seq).reshape(-1,2)
            test_label = torch.FloatTensor(test_label).reshape(-1,2)
            seq.append((test_seq, test_label))
        
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

        return seq




