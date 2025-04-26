import pandas as pd
import os
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from itertools import chain


# 检查是否有CUDA可用的GPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # 如果有GPU，使用第一个GPU
else:
    device = torch.device("cpu")  # 如果没有GPU，使用CPU


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)



#读取数据
folder_path = 'D:\\fire\data\\2500'

# 遍历文件夹中的所有文件
def get_data(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"): 
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            data.append(df)
    train = []
    # 合并训练集数据
    for i in range(len(data) - 1):
        train.append(data[i])
    # 获取最后一组数据作为测试集
    test = data[-1]
    return train, test;
# 获取训练数据和测试数据
train_data,test_data = get_data(folder_path)

# 滑动窗口处理数据
def get_train_data(data):
    print('训练集数据...')
    
    # split
    scaler = MinMaxScaler()
    print(len(data))
    for i in range(len(data)):
        x = data[i].iloc[:,1:].values
        # 归一化
        normalized_data = scaler.fit_transform(x)
        data[i] = np.column_stack((data[i].iloc[:, 0].values.reshape(-1, 1), normalized_data))


    def process(data, shuffle):
        seq = []
        for k in range(len(data)):
            for i in range(len(data[k]) - 100):
                train_seq = []
                train_label = []
                for j in range(i, i + 100):
                    x = data[k][j]
                    train_seq.append(x)
                # for c in range(2, 8):
                #     train_seq.append(data[i + 24][c])
                train_label.append(data[k][i + 100][1:])
                
                train_seq = torch.FloatTensor(train_seq)
                train_label = torch.FloatTensor(train_label).view(-1)
                seq.append((train_seq, train_label))

            # print(seq[-1])
        seq = MyDataset(seq)
        # batch_size = 10
        seq = DataLoader(dataset=seq, batch_size=10, shuffle=False, num_workers=0, drop_last=True)

        return seq

    train = process(data, False)
    return train

def get_test_data(data,len):
    print('测试集数据...')
    
    # split
    scaler = MinMaxScaler()
    print(len(data))
    x = data.iloc[:,1:].values
    normalized_data = scaler.fit_transform(x)
    data = np.column_stack((data.iloc[:, 0].values.reshape(-1, 1), normalized_data))
   
    test_seq = []
    test_label = []
   
    for i in range(len(data) - 100):
        test_label.append(data[i + 100][1:])
       
    test_label = torch.FloatTensor(test_label).view(-1)
    for j in range(0, 100):
        x = data[j]
        test_seq.append(x)
    test_seq = torch.FloatTensor(test_seq)
    return test_seq, test_label

get_train_data(train_data)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1 # 单向LSTM
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0)) # output(5, 30, 64)
        pred = self.linear(output)  # (10, 100, 37)
        pred = pred[:, -1, 1:]  # (10,100,37)
        return pred


def train(data):
    model = LSTM(input_size = 36, hidden_size = 64, num_layers = 5, output_size = 36, batch_size=10).to(device)

    loss_function = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                                     weight_decay=0.0001)
    
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    # training
    best_loss = float('inf')
    best_model = None
    count = 0
    for epoch in tqdm(range(100)):
        train_loss = []
        for (seq, label) in data:
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            if loss < best_loss:
                best_loss = loss
                count = 0
            else:
                count += 1
            if count >= 5:
                break

            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        # validation


        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        if count >= 5:
            break
        model.train()

    state = {'models': model.state_dict()}
    torch.save(state, "d:/火灾/数据/魏凯/Time_forecast/model.pth")


def test(test_seq, test_label):
    pred = []
    seq = test_seq
    y = test_label
    print('loading models...')
    model = LSTM(input_size = 1, hidden_size = 64, num_layers = 5, output_size = 1, batch_size=1).to(device)

    # models = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load("d:/火灾/数据/魏凯/Time_forecast/model.pth")['models'])
    model.eval()
    print('predicting...')
    for i in tqdm(test_label.shape[0]):
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            pred.extend(y_pred)
        seq = seq[1:]
        seq = np.insert(seq,)

    y, pred = np.array(y), np.array(pred)
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print('mape:', get_mape(y, pred))
    # plot
    x = [i for i in range(1, 151)]
    x_smooth = np.linspace(np.min(x), np.max(x), 900)
    y_smooth = make_interp_spline(x, y[150:300])(x_smooth)
    plt.plot(x_smooth, y_smooth, c='green', marker='*', ms=1, alpha=0.75, label='true')

    y_smooth = make_interp_spline(x, pred[150:300])(x_smooth)
    plt.plot(x_smooth, y_smooth, c='red', marker='o', ms=1, alpha=0.75, label='pred')
    plt.grid(axis='y')
    plt.legend()
    plt.show()
