import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import pandas as pd
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------- 1. 参数设置 ----------------------
k = 5
num_epochs = 50
lr = 0.001
weight_decay = 0.0001
batch_size = 256
image_dir = "./01_data/03_DataSet_Kaggle_leaves"

# ---------------------- 2. 自定义 Dataset ----------------------
class LeafDataset(Dataset):
    def __init__(self, img_names, labels, img_dir, transform=None):
        self.img_names = img_names
        self.labels = labels
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# ---------------------- 3. 模型定义 ----------------------
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.resnet.fc.in_features, 176)
        )

    def forward(self, x):
        return self.resnet(x)


# ---------------------- 4. 预处理函数 ----------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------- 5. 数据划分 ----------------------
def get_k_fold_data(k, i, features, labels):
    assert k > 1
    fold_size = len(features) // k
    X_train, y_train = [], []
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = features[idx], labels[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        else:
            X_train.extend(X_part)
            y_train.extend(y_part)
    return X_train, y_train, X_valid, y_valid
def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.normal_(m.weight, std=0.01)
            
# ---------------------- 6. 训练函数 ----------------------
def train(train_dataset, test_dataset, k=-1):
    net = ResNet().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    net.apply(init_weights)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("开始训练")
    for epoch in range(num_epochs):
        net.train()
        train_loss = 0.0
        correct, total = 0, 0  # 准确率统计

        for X, y in train_loader:
            X, y = X.to(device), y.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = net(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 正确率统计
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss /= len(train_loader)
        train_acc = correct / total

        # --- 测试 ---
        net.eval()
        test_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device, dtype=torch.long)
                outputs = net(X)
                loss = loss_fn(outputs, y)
                test_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        test_loss /= len(test_loader)
        test_acc = correct / total

        print(f'K: {k}, Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Acc:{train_acc:.4f}, Test Acc: {test_acc:.4f}')
        scheduler.step()

    return train_loss, test_loss, train_acc, test_acc

# ---------------------- 7. K折交叉验证 ----------------------
def train_k_fold(k):
    train_data = pd.read_csv(f"{image_dir}/train.csv")
    le = LabelEncoder()
    train_data['label'] = le.fit_transform(train_data.iloc[:, 1])  # 编码标签

    all_imgs = train_data.iloc[:, 0].values
    all_labels = train_data['label'].values
    total_train_loss, total_test_loss = 0.0, 0.0
    total_train_acc, total_test_acc = 0.0, 0.0    
    for i in range(k):
        torch.cuda.empty_cache()  # 清理显存
        train_imgs, train_labels, test_imgs, test_labels = get_k_fold_data(k, i, all_imgs, all_labels)
        train_dataset = LeafDataset(train_imgs, train_labels, image_dir, train_transform)
        test_dataset = LeafDataset(test_imgs, test_labels, image_dir, test_transform)
        train_loss, test_loss, train_acc, test_acc=train(train_dataset, test_dataset, k=i+1)
        total_train_loss += train_loss
        total_test_loss += test_loss
        total_train_acc += train_acc
        total_test_acc += test_acc
    print(f'K折交叉验证结果:\n'
          f'平均训练损失: {total_train_loss/k:.4f}, 平均测试损失: {total_test_loss/k:.4f}, '
          f'平均训练准确率: {total_train_acc/k:.4f}, 平均测试准确率: {total_test_acc/k:.4f}')
# ---------------------- 8. 运行 ----------------------
if __name__ == "__main__":
    train_k_fold(k)
