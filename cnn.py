import pandas as pd
import random
import numpy as np
from  matplotlib import pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense,SimpleRNN,LSTM
import tensorflow as tf
import os
# 只让 TensorFlow 看到 GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# TensorFlow 2.x中设置随机种子
tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)
tf.keras.utils.set_random_seed(42)

data = pd.read_csv('D:\\fire\data\\2500\\120.csv')
#print(data.head)
t = data.loc[:,'THCP']

t_norm = t/(max(t) - min(t))

# fig1 = plt.figure(figsize=(8,5))
# plt.plot(t)
# plt.title("温度变化")
# plt.xlabel("time")
# plt.ylabel("temp")
# plt.show()

def get_data(data,step):
    x = []
    y = []
    for i in range(len(data) - step):
        x.append([a for a in data[i: i + step]])
        y.append(data[i + step])
    x = np.array(x)
    x = x.reshape(x.shape[0], x.shape[1],1)
    y = np.array(y)
    y = y.reshape(y.shape[0],1)
    return x,y

x,y = get_data(t_norm,100)
print(x.shape,"====",y.shape)

model  = Sequential()
model.add(SimpleRNN(units= 5,input_shape=(100,1),activation='relu'))
model.add(Dense(units=1,activation='linear'))
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x,y,batch_size=20,epochs=50)

y_pred = model.predict(x)*(max(t) - min(t))
y = y*(max(t) - min(t))

fig1 = plt.figure(figsize=(8,5))
plt.plot(y,label="real",linewidth=2)
plt.plot(y_pred,label="predict")
plt.title("Time vs Temperature")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.legend()
plt.show()


# 读取数据
data = pd.read_csv('D:\\fire\data\\2500\\220.csv')
test_t = data.loc[:,'THCP']

test_t_norm = test_t/(max(test_t) - min(test_t))
# x_test = np.array([a for a in test_t_norm[0: 0 + 100]]).reshape(-1,100,1)
# _,y_test = get_data(test_t_norm,100)
# y_pre = []
# y_test = y_test*(max(test_t) - min(test_t))
# for i in range(len(test_t) - 100):
#     y = model.predict(x_test)*(max(test_t) - min(test_t))
#     print(y[0][0],"======")
#     y_pre.append(y[-1][0])
#     x_test = x_test[:,1:]
#     y = y.reshape(-1,y.shape[0],1)
#     x_test = np.concatenate((x_test, y), axis=1)

# #直接预测
x_test,y_test = get_data(test_t_norm,100)
y_pre = model.predict(x_test)*(max(test_t) - min(test_t))
y_test = y_test*(max(test_t) - min(test_t))

fig2 = plt.figure(figsize=(8,5))
plt.plot(y_test,label="real",linewidth=2)
plt.plot(y_pre,label="predict")
plt.title("Time vs Temperature")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.legend()
plt.show()

result_test = np.array(y_test).reshape(-1,1)
result_pre = np.array(y_pre).reshape(-1,1)
result = np.concatenate( (result_test, result_pre), axis=1)
result = pd.DataFrame(data=result,columns=['real','predict'])
result.to_csv('D:\\fire\data\\220_predict.csv',index=False)