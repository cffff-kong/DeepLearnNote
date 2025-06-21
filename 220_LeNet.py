import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 256
lr = 0.9
num_epochs = 100

# 数据加载
train_iter = DataLoader(
    datasets.FashionMNIST(root='01_data/01_DataSet_FashionMNIST', train=True, download=True,
                        transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

test_iter = DataLoader(
    datasets.FashionMNIST(root='01_data/01_DataSet_FashionMNIST', train=False, download=True,
                        transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=False)

class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.net = torch.nn.Sequential(
            Reshape(),
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10)
        )
    
    def forward(self, x):
        return self.net(x)

def train_epoch(net, train_iter, test_iter, num_epochs):
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight) # 根据输入、输出大小，使得随即初始化后，输入和输出的的方差是差不多的 
    for epoch in range(num_epochs):
        net.train()
        train_loss, train_acc = 0.0, 0.0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = net(X)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += (output.argmax(1) == y).sum().item()
        
        # 计算训练集平均损失和准确率
        train_loss /= len(train_iter)
        train_acc /= len(train_iter.dataset)
        
        # 测试集评估
        net.eval()
        test_loss, test_acc = 0.0, 0.0
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                output = net(X)
                test_loss += loss_fn(output, y).item()
                test_acc += (output.argmax(1) == y).sum().item()
        
        test_loss /= len(test_iter)
        test_acc /= len(test_iter.dataset)
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

# 初始化网络并训练
net = LeNet().to(device)
train_epoch(net, train_iter, test_iter, num_epochs)