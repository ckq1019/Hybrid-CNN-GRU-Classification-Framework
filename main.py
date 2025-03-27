import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, ConvLSTM1D, BatchNormalization, GRU, TimeDistributed, Conv1D, MaxPool1D
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import time
import torch.nn as nn
import torch
import math


# 导入数据
def load_data(file_path) -> pd.DataFrame:
    if not os.path.exists(file_path):
        print("请检查文件是否存在！")
        return None
    df = pd.read_csv(file_path)
    return df


# MinMaxScaler数据归一化，可以帮助网络模型更快的拟合，稍微有一些提高准确率的效果
def minmaxscaler(data: pd.DataFrame) -> pd.DataFrame:
    volume = data.BEKES.values
    volume = volume.reshape(len(volume), 1)
    volume = scaler.fit_transform(volume)
    volume = volume.reshape(len(volume), )
    for place_name in data.columns:
        if place_name == 'Date':
            continue
        data[place_name] = volume
    return data


# 划分训练集和测试集
def split_data(train_data, test_data, n_val, n_in, n_out):
    x_train = train_data[:, n_val:].T
    y_train = test_data[:, n_val:].T
    x_val = train_data[:, :n_val].T
    y_val = test_data[:, :n_val].T
    return x_train, y_train, x_val.reshape((n_val, n_in)), y_val.reshape((n_val, n_out))


# 构建LSTM
def build_lstm(n_in, n_out):
    model = Sequential()
    model.add(LSTM(256, activation='relu', input_shape=(n_in, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(n_out))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mae')
    return model


# 构建CovLSTM
def build_covlstm(n_in, n_out):
    model = Sequential()
    model.add(ConvLSTM1D(filters=n_out, kernel_size=3, input_shape=(1, n_in, 1), return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(n_out))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mae')
    return model


# GRU模型
def build_gru(n_in, n_out):
    model = Sequential()
    model.add(GRU(19, input_shape=(n_in, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(19, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed((Dense(n_out))))
    model.add(Flatten())
    model.add((Dense(n_out)))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mae')
    return model


# CNN-LSTM模型
def build_cnnlstm(n_in, n_out):
    model = Sequential()
    model.add(Conv1D(filters=19, kernel_size=19, padding='same', activation='relu'))
    model.add(MaxPool1D(pool_size=2, strides=1, padding='same'))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(n_out))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mae')
    return model


# CNN-GRU模型
def build_cnn_gru(n_in, n_out):
    model = Sequential()
    model.add(Conv1D(filters=19, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPool1D(pool_size=2, strides=1, padding='same'))
    model.add(Dropout(0.2))
    model.add(GRU(19, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(19, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed((Dense(n_out))))
    model.add(Flatten())
    model.add((Dense(n_out)))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mae')
    return model


# 模型训练
def model_fit(model, x_train, y_train, x_val, y_val):
    model.fit(x=x_train, y=y_train, epochs=100, batch_size=19, verbose=1, validation_data=(x_val, y_val))
    m = model.evaluate(x_val, y_val)
    print(m)
    return model


if __name__ == '__main__':
    # 读入数据
    file_path_train = os.path.join(os.getcwd(), "Data", "pre_423days_train_data.csv")  # 训练数据集合路径
    file_path_test = os.path.join(os.getcwd(), "Data", "last_100days_true_data.csv")  # 测试数据集合路径
    train_data = load_data(file_path_train)  # 从文件中读取数据
    test_data = load_data(file_path_test)

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = minmaxscaler(train_data)
    test_data = minmaxscaler(test_data)

    n_in = 422  # 输入步长
    n_out = 100  # 预测步长
    n_val = 1  # 测试数据集个数
    x_train, y_train, x_val, y_val = split_data(np.array(train_data.iloc[:, 1:], dtype=np.float64),
                                                np.array(test_data.iloc[:, 1:], dtype=np.float64), n_val, n_in, n_out)

    # 1.用LSTM进行预测
    model = build_lstm(n_in, n_out)
    start_time = time.time()
    model_fit(model, x_train, y_train, x_val, y_val)  # 训练模型
    end_time = time.time()
    # 模型预测
    predict = model.predict(x_val)
    predict = scaler.inverse_transform(predict)[0]  # 还原数据
    actual = scaler.inverse_transform(y_val)[0]  # 还原数据

    # 预测结果图展示
    x = [x for x in range(n_out)]
    fig, ax = plt.subplots(figsize=(15, 5), dpi=300)
    ax.plot(x, predict, linewidth=2.0, label="predict")
    ax.plot(x, actual, linewidth=2.0, label="actual")
    ax.legend(loc=2)
    plt.title("LSTM predict")
    plt.ylim((0, 70))
    plt.grid(linestyle='-.')
    plt.show()

    from sklearn import metrics
    # 结果评估
    print('Spend Time:', end_time - start_time)
    print('MAE of LSTM:', metrics.mean_absolute_error(actual, predict))
    print('MSE of LSTM:', metrics.mean_squared_error(actual, predict))

    # 2.用CovLSTM进行预测
    x_train = x_train.reshape((-1, 1, n_in))
    y_train = y_train.reshape((-1, 1, n_out))
    x_val = x_val.reshape((-1, 1, n_in))
    y_val = y_val.reshape((-1, 1, n_out))
    model = build_covlstm(n_in, n_out)
    start_time = time.time()
    model_fit(model, x_train, y_train, x_val, y_val)  # 训练模型
    end_time = time.time()
    # 模型预测
    predict = model.predict(x_val)
    predict = scaler.inverse_transform(predict.reshape((1, -1)))[0]  # 还原数据
    actual = scaler.inverse_transform(y_val.reshape(1, -1))[0]  # 还原数据

    # 预测结果图展示
    x = [x for x in range(n_out)]
    fig, ax = plt.subplots(figsize=(15, 5), dpi=300)
    ax.plot(x, predict, linewidth=2.0, label="predict")
    ax.plot(x, actual, linewidth=2.0, label="actual")
    ax.legend(loc=2)
    plt.title("CovLSTM predict")
    plt.ylim((0, 70))
    plt.grid(linestyle='-.')
    plt.show()

    from sklearn import metrics

    # 结果评估
    print('Spend Time:', end_time - start_time)
    print('MAE of CovLSTM:', metrics.mean_absolute_error(actual, predict))
    print('MSE of CovLSTM:', metrics.mean_squared_error(actual, predict))

    # 3.用GRU模型预测
    x_train = x_train.reshape((-1, n_in, 1))
    y_train = y_train.reshape((-1, n_out, 1))
    x_val = x_val.reshape((-1, n_in, 1))
    y_val = y_val.reshape((-1, n_out, 1))
    model = build_gru(n_in, n_out)
    start_time = time.time()
    model_fit(model, x_train, y_train, x_val, y_val)  # 训练模型
    end_time = time.time()
    # 模型预测
    predict = model.predict(x_val)
    predict = scaler.inverse_transform(predict.reshape(1, -1))[0]  # 还原数据
    actual = scaler.inverse_transform(y_val.reshape(1, -1))[0]  # 还原数据

    # 预测结果图展示
    x = [x for x in range(n_out)]
    fig, ax = plt.subplots(figsize=(15, 5), dpi=300)
    ax.plot(x, predict, linewidth=2.0, label="predict")
    ax.plot(x, actual, linewidth=2.0, label="actual")
    ax.legend(loc=2)
    plt.title("GRU predict")
    plt.ylim((0, 70))
    plt.grid(linestyle='-.')
    plt.show()

    from sklearn import metrics

    # 结果评估
    print('Spend Time:', end_time - start_time)
    print('MAE of GRU:', metrics.mean_absolute_error(actual, predict))
    print('MSE of GRU:', metrics.mean_squared_error(actual, predict))

    # 4.用CNN-LSTM模型预测
    x_train = x_train.reshape((-1, n_in, 1))
    y_train = y_train.reshape((-1, n_out, 1))
    x_val = x_val.reshape((-1, n_in, 1))
    y_val = y_val.reshape((-1, n_out, 1))
    model = build_cnnlstm(n_in, n_out)
    start_time = time.time()
    model_fit(model, x_train, y_train, x_val, y_val)  # 训练模型
    end_time = time.time()
    # 模型预测
    predict = model.predict(x_val)
    predict = scaler.inverse_transform(predict.reshape(1, -1))[0]  # 还原数据
    actual = scaler.inverse_transform(y_val.reshape(1, -1))[0]  # 还原数据

    # 预测结果图展示
    x = [x for x in range(n_out)]
    fig, ax = plt.subplots(figsize=(15, 5), dpi=300)
    ax.plot(x, predict, linewidth=2.0, label="predict")
    ax.plot(x, actual, linewidth=2.0, label="actual")
    ax.legend(loc=2)
    plt.title("CNN-LSTM predict")
    plt.ylim((0, 70))
    plt.grid(linestyle='-.')
    plt.show()

    from sklearn import metrics

    # 结果评估
    print('Spend Time:', end_time - start_time)
    print('MAE of CNN-LSTM:', metrics.mean_absolute_error(actual, predict))
    print('MSE of CNN-LSTM:', metrics.mean_squared_error(actual, predict))

    # 4.用CNN-GRU模型预测
    x_train = x_train.reshape((-1, n_in, 1))
    y_train = y_train.reshape((-1, n_out, 1))
    x_val = x_val.reshape((-1, n_in, 1))
    y_val = y_val.reshape((-1, n_out, 1))
    model = build_cnn_gru(n_in, n_out)
    start_time = time.time()
    model_fit(model, x_train, y_train, x_val, y_val)  # 训练模型
    end_time = time.time()
    # 模型预测
    predict = model.predict(x_val)
    predict = scaler.inverse_transform(predict.reshape(1, -1))[0]  # 还原数据
    actual = scaler.inverse_transform(y_val.reshape(1, -1))[0]  # 还原数据

    # 预测结果图展示
    x = [x for x in range(n_out)]
    fig, ax = plt.subplots(figsize=(15, 5), dpi=300)
    ax.plot(x, predict, linewidth=2.0, label="predict")
    ax.plot(x, actual, linewidth=2.0, label="actual")
    ax.legend(loc=2)
    plt.title("CNN-GRU predict")
    plt.ylim((0, 70))
    plt.grid(linestyle='-.')
    plt.show()

    from sklearn import metrics

    # 结果评估
    print('Spend Time:', end_time - start_time)
    print('MAE of CNN-GRU:', metrics.mean_absolute_error(actual, predict))
    print('MSE of CNN-GRU:', metrics.mean_squared_error(actual, predict))

