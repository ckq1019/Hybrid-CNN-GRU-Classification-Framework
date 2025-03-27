一、背景

利用模型对水痘空气传播疾病时间序列数据进行预测。

二、需求分析

1.数据处理需求：需要对 2005 年 1 月至 2015 年 1 月匈牙利 20 个城市（19 个县和首都布达佩斯）的水痘病例数据进行处理，包括数据读取、划分训练集和测试集、数据归一化等操作，为模型训练做准备。

2.模型构建与比较需求：构建 CNN-GRU 模型，并与 GRU、LSTM、CovLSTM、CNN-LSTM 等模型进行对比，通过实验评估不同模型在预测匈牙利水痘病例时间序列上的性能。

3.评估指标需求：确定均方误差（MSE）、平均绝对误差（MAE）和训练模型时间作为评估指标，用于衡量模型预测的准确性和训练效率。

三、关键技术

1. 深度学习框架：使用 Keras 搭建模型，利用其简洁的 API 快速构建各种神经网络模型，如 LSTM、GRU、CNN-LSTM、CNN-GRU 等。
   
3. 数据处理库：借助 Pandas 进行数据读取和预处理，将原始数据转换为适合模型输入的格式；利用 MinMaxScaler 对数据进行归一化处理，提升模型训练效果。
   
5. 神经网络层：运用卷积层（Conv1D）进行特征提取，捕捉数据的局部特征；使用池化层（MaxPool1D）降低数据维度，减少计算量；通过循环层（LSTM、GRU）处理时间序列数据，学习数据的时间依赖性；利用全连接层（Dense）进行最终的预测输出。
   
7. 优化器：采用 Adam 优化器，通过自适应调整学习率，加快模型的收敛速度，提高训练效率。
   
四、目标

1.构建有效模型：构建基于 CNN-GRU 的时间序列预测模型，能够准确预测匈牙利水痘病例的数量变化趋势。

2.对比评估模型：对比 CNN-GRU 模型与其他经典时间序列模型（GRU、LSTM、CovLSTM、CNN-LSTM）的性能，分析各模型的优劣。

3.提高预测性能：提高模型预测的准确性和可扩展性，为水痘疾病的预防和研究提供更可靠的支持。

五、成果

1.模型性能优势：实验结果表明，CNN-GRU 模型在均方误差（MSE）和平均绝对误差（MAE）指标上远优于 LSTM、CovLSTM 和 CNN-LSTM 模型，与 GRU 模型相比，CNN-GRU 模型的结果也更好，预测准确性更高。

2.训练效率优势：在训练时间方面，CNN-GRU 模型和 GRU 模型远快于 LSTM、CovLSTM 和 CNN-LSTM 模型，虽然 CNN-GRU 模型因多一层卷积层比 GRU 模型稍慢，但整体效率仍较高。

3.预测效果可视化：通过绘制预测结果图，直观展示了 CNN-GRU 模型预测结果与真实结果基本一致，进一步验证了模型的有效性。

![image](https://github.com/user-attachments/assets/7a46bf86-70c7-4e22-8643-19894e30bc32)

![Uploading image.png…]()


![image](https://github.com/user-attachments/assets/ffc698a4-bb4a-4405-a651-302d53032c55)

![image](https://github.com/user-attachments/assets/e18384fc-e3b2-4ad7-9c46-67c43c5f2b41)

![image](https://github.com/user-attachments/assets/c8f9cd81-aaff-41fd-9b7a-f21d9ea5577b)
