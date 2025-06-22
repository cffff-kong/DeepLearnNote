import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 500
lr = 0.01
num_epochs = 100

# 数据加载
transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),  # AlexNet通常使用224x224的输入尺寸
])
train_iter = DataLoader(
    datasets.FashionMNIST(root='01_data/01_DataSet_FashionMNIST', train=True, download=True,
                        transform=transform),batch_size=batch_size, shuffle=True)

test_iter = DataLoader(
    datasets.FashionMNIST(root='01_data/01_DataSet_FashionMNIST', train=False, download=True,
                        transform=transform),batch_size=batch_size, shuffle=False)
print("Data loaded successfully.")
class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.net = torch.nn.Sequential(
        nn.Conv2d(1,96,kernel_size=11,stride=4,padding=1),nn.ReLU(), # 数据集为fashion_mnist图片，所以输入通道为1，如果是Imagnet图片，则通道数应为3     
        nn.MaxPool2d(kernel_size=3,stride=2),
        nn.Conv2d(96,256,kernel_size=5,padding=2),nn.ReLU(), # 256为输出通道数
        nn.MaxPool2d(kernel_size=3,stride=2),
        nn.Conv2d(256,384,kernel_size=3,padding=1),nn.ReLU(),
        nn.Conv2d(384,384,kernel_size=3,padding=1),nn.ReLU(),
        nn.Conv2d(384,256,kernel_size=3,padding=1),nn.ReLU(),
        nn.MaxPool2d(kernel_size=3,stride=2),nn.Flatten(),
        nn.Linear(6400,4096),nn.ReLU(),nn.Dropout(p=0.5),
        nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(p=0.5),
        nn.Linear(4096,10))
    
    def forward(self, x):
        return self.net(x)

def train_epoch(net, train_iter, test_iter, num_epochs):
    train_num=0
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
            train_num+=1
            if train_num % 20 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Step {train_num}, Loss: {train_loss:.4f}')
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
        train_num=0
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

# 初始化网络并训练
net = AlexNet().to(device)
train_epoch(net, train_iter, test_iter, num_epochs)