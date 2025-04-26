import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential,Model
from keras.layers import Dense,SimpleRNN,LSTM,Dropout,Embedding,Input
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.layers import Dense,Input, Dropout, Embedding, Flatten,MaxPooling1D,Conv1D,SimpleRNN,LSTM,GRU,Multiply,GlobalMaxPooling1D
from model import *
import argparse
import os
from keras.layers import Bidirectional,Activation,BatchNormalization,GlobalAveragePooling1D,MultiHeadAttention
from keras.models import load_model

# TensorFlow 2.x中设置随机种子
tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)
tf.keras.utils.set_random_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--choose_model', type=str, required=True, default='LSTM',
                    help='model name, options: [LSTM, BiLSTM, CNN+LSTM]')
parser.add_argument('--folder_path', type=str, required=True, default='D:/fire/data/2500/',
                    help='path of the data folder')


args = parser.parse_args()

choose_model = args.choose_model
#读取数据
folder_path = args.folder_path

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    #return 0.01*(u / d).mean(-1)
    return 0.01*(u / d)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr







time = []
# 遍历文件夹中的所有文件
# def get_data(folder_path):
#     global time
#     train = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".csv"):
#             # 训练集                 
#             file_path = os.path.join(folder_path, filename)
#             df = pd.read_csv(file_path,dtype=float)
#             df = df.set_index('Time').sort_index()
#             if(filename != 'test.csv'):
#                 train.append(df)
#             else:
#                 test = df
#     time = test.index
#     return train, test;
# 获取训练数据和测试数据


def get_data(folder_path):
    global time
    train = []
    test = []
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # 按照文件名进行排序（字母顺序）
    csv_files.sort()
    for filename in csv_files:
        if filename.endswith(".csv"):
            # 训练集                 
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path,dtype=float)
            df = df.set_index('Time').sort_index()
            # 训练集
            if('test' not in filename):
                train.append(df)
            # 测试集    
            else:
                test.append(df)
    time = train[0].index
    return train, test;

train_data,test_data = get_data(folder_path)


# 滑动窗口处理数据
def get_train_data(data):
    print('训练集数据...')
    
    # split
    scaler = MinMaxScaler()
    print(len(data))
    train_seq = []
    train_label = []
    for i in range(len(data)):
        x = data[i]
        # 归一化
        normalized_data = scaler.fit_transform(x)
        for j in range(0,18):
            for i in range(len(normalized_data) - 150):
                for k in range(i,i+100):
                    # 温度加顶棚温度
                    train_seq.append([normalized_data[k,j],normalized_data[k,-1]])
                train_label.append([normalized_data[i + 150,j], normalized_data[i + 150,-1]])
    
    train_seq = np.array(train_seq)
    train_seq = train_seq.reshape(-1,100,2)
    print(train_seq.shape)
    train_label = np.array(train_label)
    train_label = train_label.reshape(-1,2)
    print(train_label.shape)
    return train_seq,train_label

def get_test_data(data):
    print('测试集数据...')
    
    # split
    scaler = MinMaxScaler()
    test_seq = []
    test_label = []
    x = data
    # 归一化
    normalized_data = scaler.fit_transform(x)
    for i in range(len(normalized_data) - 150):
        for k in range(i,i+100):
            #test_seq.append([normalized_data[k,0],normalized_data[k,-1]])
            #第二个测点
            test_seq.append([normalized_data[k,10],normalized_data[k,-1]])
        #test_label.append([normalized_data[i + 150,0], normalized_data[i + 150,-1]])
        test_label.append([normalized_data[i + 150,10], normalized_data[i + 150,-1]])
    
    test_seq = np.array(test_seq)
    #数据格式 [batch_size,len,dim]
    test_seq = test_seq.reshape(-1, 100 ,2)
    test_label = np.array(test_label)
    # [len,dim]
    test_label = test_label.reshape(-1,2)

    return test_seq,test_label



# x_train, y_train = get_train_data(train_data)

#  网络架构
# return_sequences返回所有时序的值，false是返回最后一个
# x_train  [batch_size,len,dim]


best_model_path = folder_path + choose_model  + '/model/' + 'my_model.h5'
check_path = os.path.join(folder_path, choose_model, 'model', 'my_model.h5')
# 检查路径的父目录是否存在，如果不存在则创建它
dir_path = os.path.dirname(check_path)


#test部分
# 加载模型,
print('test >> loading model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
model = load_model(best_model_path)

#door
dict = ['60','100','220']


# water
# dict = ['120','220','340','120','220','340','120','220','340']
# dict_dir = [20,20,20,50,50,50,80,80,80]

# exhaust
#dict = ['120','180','300','120','180','300','120','180','300']
#dict_dir = [2,2,2,6,6,6,10,10,10]
j = -1
for i in range(len(test_data)):
    j = j + 1
    x_test, y_test = get_test_data(test_data[i])
    print(x_test.shape)
    y_pre_all = model.predict(x_test)

    # 温度
    #min_val = test_data[i].iloc[:,0].min()
    #max_val = test_data[i].iloc[:,0].max()
    #测点2
    min_val = test_data[i].iloc[:,10].min()
    max_val = test_data[i].iloc[:,10].max()
    yy_pre = y_pre_all[:,0]*(max_val-min_val)+min_val
    yy = y_test[:,0]*(max_val-min_val)+min_val
    
    # 顶棚温度
    min_val_t = test_data[i].iloc[:,-1].min()
    max_val_t = test_data[i].iloc[:,-1].max()
    yy_pre_t = y_pre_all[:,-1]*(max_val_t-min_val_t)+min_val_t
    yy_t = y_test[:,-1]*(max_val_t-min_val_t)+min_val_t

    # 温度
    preds = y_pre_all[:,0]
    trues = y_test[:,0]
    
    #顶棚温度
    preds_t = y_pre_all[:,-1]
    trues_t = y_test[:,-1]
    # result save
    # folder_path = './results/' + setting + '/'
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)

    mae, mse, rmse, mape, mspe, rse, corr = metric(yy, yy_pre)
    mae_t, mse_t, rmse_t, mape_t, mspe_t, rse_t, corr_t = metric(yy_t, yy_pre_t)

    data_rows = [{'time': t, 'Real': r, 'Predicted Value': p} for t, r, p in zip(time[-len(yy):], yy, yy_pre)]
    print("############################################################################")
    # 将字典列表转换为DataFrame
    df = pd.DataFrame(data_rows)
    # 将DataFrame保存到CSV文件
    df.to_csv(folder_path + choose_model +'/senior2.csv', index=False)

    # data_rows = [{'time': t, 'Real': r, 'Predicted Value': p} for t, r, p in zip(time[-len(yy):], yy_t, yy_pre_t)]
    # # 将字典列表转换为DataFrame
    # df = pd.DataFrame(data_rows)
    # # 将DataFrame保存到CSV文件
    # df.to_csv(folder_path + choose_model + '/ceiling.csv', index=False)



    print('==========  MAPE:{}, mae:{}'.format(mape, mae))
    f = open(folder_path + choose_model + "/result.txt", 'a')
    f.write(dict[i] + "new >> senior2_temperature>>>>>>>>>>>>>>>>>>>>>>." + "  \n")
    #f.write(str(dict_dir[j]) + '##' + str(dict[j]) + ">>>>>>>>>>>>>>>>>>>>>>." + "  \n")
    f.write('mse:{}, mae:{}, rmse:{},mape:{},mspe:{},rse:{}, corr:{}'.format(mse, mae,rmse, mape, mspe, rse, corr))
    # f.write("\n" + "new >>> ceiling temperature>>>>>>>>>>>>>>>>>>>>>>." + "  \n")
    # f.write('mse:{}, mae:{}, rmse:{},mape:{},mspe:{},rse:{}, corr:{}'.format(mse_t, mae_t,rmse_t, mape_t, mspe_t, rse_t, corr_t))
    # # f.write('\n')
    f.write('\n')
    f.close()


    # if '120' in dict[i]:
    #     fig2 = plt.figure(figsize=(8,5))
    #     plt.plot(time[-len(yy):],yy,label="real",color='red',linewidth=1.5)
    #     plt.plot(time[-len(yy):],yy_pre,label="predict",color='blue',linewidth=1.5)
    #     plt.xlabel("Time(s)")
    #     plt.ylabel("Temperature(℃)")
    #     plt.legend()
    #     plt.grid(linestyle='-.',alpha=0.3)
    #     ax = plt.gca()
    #     xx = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.55
    #     yy = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.96
    #     formatted_text = 'MAPE = {:.2f}%\nMAE = {:.2f}'.format(mape * 100, mae)
    #     plt.text(xx, yy, formatted_text, fontsize=13,ha='left', va='top')
    #     plt.savefig(folder_path + choose_model + "/img/" + dict[i]+".png")

    # else:
    #     fig2 = plt.figure(figsize=(8,5))
    #     plt.plot(time[-len(yy):],yy,label="real",color='red',linewidth=1.5)
    #     plt.plot(time[-len(yy):],yy_pre,label="predict",color='blue',linewidth=1.5)
    #     plt.xlabel("Time(s)")
    #     plt.ylabel("Temperature(℃)")
    #     plt.legend()
    #     plt.grid(linestyle='-.',alpha=0.3)
    #     ax = plt.gca()
    #     xx = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02
    #     yy = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.96
    #     formatted_text = 'MAPE = {:.2f}%\nMAE = {:.2f}'.format(mape * 100, mae)
    #     plt.text(xx, yy, formatted_text, fontsize=13,ha='left', va='top')
    #     plt.savefig(folder_path + choose_model + "/img/" + dict[i]+".png")


    # fig2 = plt.figure(figsize=(8,5))
    # plt.plot(time[-len(yy):],yy,label="real",color='red',linewidth=1.5)
    # plt.plot(time[-len(yy):],yy_pre,label="predict",color='blue',linewidth=1.5)
    # plt.xlabel("Time(s)")
    # plt.ylabel("Temperature(℃)")
    # plt.legend()
    # plt.grid(linestyle='-.',alpha=0.3)
    # ax = plt.gca()
    # xx = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.55
    # yy = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.96
    # formatted_text = 'MAPE = {:.2f}%\nMAE = {:.2f}'.format(mape * 100, mae)
    # plt.text(xx, yy, formatted_text, fontsize=13,ha='left', va='top')
    # plt.savefig(folder_path + choose_model + "/img/" + str(dict_dir[j]) + "_" + str(dict[j])+".png")

    # fig3 = plt.figure(figsize=(8,5))
    # plt.plot(time[-len(yy_t):],yy_t,label="real",color='red',linewidth=1.5)
    # plt.plot(time[-len(yy_t):],yy_pre_t,label="predict",color='blue',linewidth=1.5)
    # plt.xlabel("Time(s)")
    # plt.ylabel("Ceiling Temperature(℃)")
    # plt.legend()
    # plt.grid(linestyle='-.',alpha=0.3)
    # ax = plt.gca()
    # xx = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.55
    # yy = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.96
    # formatted_text = 'MAPE = {:.2f}%\nMAE = {:.2f}'.format(mape_t * 100, mae_t)
    # plt.text(xx, yy, formatted_text, fontsize=13,ha='left', va='top')
    # plt.savefig(folder_path + choose_model + "/img/" + str(dict_dir[j]) + "_" + str(dict[j]) + 'ceiling' + ".png")