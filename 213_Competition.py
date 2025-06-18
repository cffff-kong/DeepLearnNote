import hashlib
import os
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_data=pd.read_csv("./01_data/02_DataSet_Kaggle_House/test.csv")

if os.path.exists("./01_data/02_DataSet_Kaggle_House/cached_test_features.pt"):
    # 如果已经存在缓存文件，则直接加载
    train_features = torch.load("./01_data/02_DataSet_Kaggle_House/cached_train_features.pt")
    test_features = torch.load("./01_data/02_DataSet_Kaggle_House/cached_test_features.pt")
    train_labels = torch.load("./01_data/02_DataSet_Kaggle_House/cached_train_labels.pt")
    # 将数据移至 GPU
    train_features = train_features.to(device)
    test_features = test_features.to(device)
    train_labels = train_labels.to(device)
    print("加载咯")
    print(test_features.shape) # 1460个样本，1个label
else:
    train_data=pd.read_csv("./01_data/02_DataSet_Kaggle_House/train.csv")
    last_column = train_data.iloc[:, -1]
    all_features = pd.concat((
        train_data.drop(columns=['Sold Price']).iloc[:, 1:],  # 去掉 'Sold price'，再从第2列开始
        test_data.iloc[:, 1:]  # 从第2列开始
    ))

    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index  # 当值的类型不是object的话，就是一个数值

    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std())) # 对数值数据变为总体为均值为0，方差为1的分布的数据        
    all_features[numeric_features] = all_features[numeric_features].fillna(0)  # 将数值数据中not number的数据用0填充  # 处理离散值。用一次独热编码替换它们
    
    # 找出 object 列中 取值数 ≤ 10 的列
    object_cols = all_features.dtypes[all_features.dtypes == 'object'].index
    low_card_cols = [col for col in object_cols if all_features[col].nunique() <= 10]

    # 丢掉取值数太多的 object 列
    all_features = all_features.drop(columns=[col for col in object_cols if col not in low_card_cols])

    # 对剩下的 low-cardinality 类别做 one-hot 编码
    all_features = pd.get_dummies(all_features, dummy_na=True)

    n_train = train_data.shape[0] # 样本个数
    train_features = torch.tensor(all_features[:n_train].values.astype(np.float32))
    test_features = torch.tensor(all_features[n_train:].values.astype(np.float32))

    # train_data的Sold price列是label值
    train_labels = torch.tensor(train_data["Sold Price"].values.reshape(-1, 1),
                            dtype=torch.float32)
    print(train_labels.shape) # 1460个样本，1个label
    # 缓存保存
    torch.save(train_features, "./01_data/02_DataSet_Kaggle_House/cached_train_features.pt")
    torch.save(test_features, "./01_data/02_DataSet_Kaggle_House/cached_test_features.pt")
    torch.save(train_labels, "./01_data/02_DataSet_Kaggle_House/cached_train_labels.pt")

# 训练相关
loss = nn.MSELoss()
print(train_features.shape[1]) # 所有特征个数
in_features = train_features.shape[1]
def get_net():
    net = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    ).to(device)  # 将模型移至GPU
    return net


def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features),1,float('inf')) # 把模型输出的值限制在1和inf之间，inf代表无穷大（infinity的缩写）       
    rmse = torch.sqrt(loss(torch.log(clipped_preds),torch.log(labels))) # 预测做log，label做log，然后丢到MSE损失函数里
    return rmse.item()

#函数将借助Adam优化器
def train(net, train_features, train_labels, test_features, test_labels,
         num_epochs, learning_rate, weight_decay, batch_size, k):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    for epoch in range(num_epochs):
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)  # 将批次数据移至 GPU
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net,train_features,train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))

        print(epoch + 1, 'epoch:', f'train log rmse {float(train_ls[-1]):f}')
    return train_ls, test_ls

# K折交叉验证
def get_k_fold_data(k,i,X,y): # 给定k折，给定第几折，返回相应的训练集、测试集
    assert k > 1
    fold_size = X.shape[0] // k  # 每一折的大小为样本数除以k
    X_train, y_train = None, None
    for j in range(k): # 每一折
        idx = slice(j * fold_size, (j+1)*fold_size) # 每一折的切片索引间隔  
        X_part, y_part = X[idx,:], y[idx] # 把每一折对应部分取出来
        if j == i: # i表示第几折，把它作为验证集
            X_valid, y_valid = X_part, y_part
        elif X_train is None: # 第一次看到X_train，则把它存起来 
            X_train, y_train = X_part, y_part
        else: # 后面再看到，除了第i外，其余折也作为训练数据集，用torch.cat将原先的合并    
            X_train = torch.cat([X_train, X_part],0)
            y_train = torch.cat([y_train, y_part],0)
    return X_train, y_train, X_valid, y_valid # 返回训练集和验证集

# 返回训练和验证误差的平均值
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train) # 把第i折对应分开的数据集、验证集拿出来   
        net = get_net()
        # *是解码，变成前面返回的四个数据
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size,k) # 训练集、验证集丢进train函数 
        
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls,valid_ls],
                    xlabel='epoch',ylabel='rmse',xlim=[1,num_epochs],
                    legend=['train','valid'],yscale='log')
        print(f'fold{i+1},train log rmse {float(train_ls[-1]):f},'
             f'valid log rmse {float (valid_ls[-1]):f}')
    return  train_l_sum / k, valid_l_sum / k # 求和做平均

# 模型选择
k, num_epochs, lr, weight_decay, batch_size = 5,500, 0.05, 0.05, 256


#train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)   
#print(f'{k}-折验证：平均训练log rmse：{float(train_l):f},'f'平均验证log rmse：{float(valid_l):f}')    

def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size,k)
    print(f'train log rmse {float(train_ls[-1]):f}')
    
    # 预测测试集
    # 预测时移动测试数据到 GPU
    preds = net(test_features.to(device)).detach().cpu().numpy()  # 预测完移回 CPU    
    # 保存预测结果（假设 test_data 有 'Id' 列）
    test_data['Sold Price'] = pd.Series(preds.reshape(1,-1)[0])

    submission = pd.concat([test_data['Id'],test_data['Sold Price']],axis=1)
    submission.to_csv('submission.csv', index=False)
    
train_and_pred(train_features, test_features, train_labels, test_data,
              num_epochs, lr, weight_decay, batch_size)