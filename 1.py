import matplotlib.pyplot as plt
import numpy as np
from keras import layers
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from datetime import datetime, timedelta


x = [1, 2, 3, 4]
y = [2, 4, 8, 16]
y2 = [4, 16, 64, 256]
x_text = 2.15
y_text = 16.513

# # 解决中文乱码问题
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.xlabel('时间(S)')
# plt.ylabel('细菌数量(℃)')
# plt.grid(linestyle='-.',alpha=0.3)

# plt.plot(x, y, label='细菌1生长曲线', color='red', linewidth=1)
# plt.plot(x, y2, label='细菌2生长曲线', color='blue', linewidth=1)
# # 图例
# plt.legend()
# ax = plt.gca()
# xx = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02
# yy = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.8
# print(xx, yy)

# formatted_text = 'x_text = {:.2f}%\ny_text = {:.2f}'.format(x_text * 100, y_text)
# print(formatted_text)
# plt.text(xx, yy, formatted_text, fontsize=12,ha='left', va='top')
# plt.show()

folder_path = "./data/door_pre4055/"
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
                print('train',filename)
            # 测试集    
            else:
                test.append(df)
                print('test',filename)
    time = train[0].index
    return train, test;

train_data,test_data = get_data(folder_path)


