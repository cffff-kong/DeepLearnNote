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
num_epochs = 30
lr = 0.0001
weight_decay = 0.0001
batch_size = 32
train_image_dir = "./01_data/04_DataSet_Kaggle_CIFAR10/train"
test_image_dir = "./01_data/04_DataSet_Kaggle_CIFAR10/test"
dataset_dir="./01_data/04_DataSet_Kaggle_CIFAR10"

# ---------------------- 2. 自定义 Dataset ----------------------
class CIFAR10Dataset(Dataset):
    def __init__(self, img_names, labels, img_dir, transform=None):
        self.img_names = [str(x) + ".png" for x in img_names]
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
            nn.Linear(self.resnet.fc.in_features, 10)
        )

    def forward(self, x):
        return self.resnet(x)
    
# ---------------------- 4. 预处理函数 ----------------------
train_transform = transforms.Compose([
    transforms.Resize(40),
    transforms.RandomResizedCrop(32,scale=(0.64,1.0),ratio=(1.0,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914,0.4822,0.4465],
                                   [0.2023,0.1994,0.2010])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914,0.4822,0.4465],
                                    [0.2023,0.1994,0.2010])
])
# ---------------------- 5. 训练函数 ----------------------
def train(train_dataset, test_dataset, k=-1):
    net = ResNet().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

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
        scheduler.step()

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

    return train_loss, test_loss, train_acc, test_acc

# ---------------------- 6. K折交叉验证 ----------------------
from sklearn.model_selection import StratifiedKFold

def train_k_fold(k):
    train_data = pd.read_csv(f"{dataset_dir}/trainLabels.csv")
    le = LabelEncoder()
    train_data['label'] = le.fit_transform(train_data.iloc[:, 1])  # 标签编码
    
    all_imgs = train_data.iloc[:, 0].values
    all_labels = train_data['label'].values

    total_train_loss, total_test_loss = 0.0, 0.0
    total_train_acc, total_test_acc = 0.0, 0.0    

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_imgs, all_labels)):
        torch.cuda.empty_cache()  # 清理显存

        # 划分数据
        train_imgs, val_imgs = all_imgs[train_idx], all_imgs[val_idx]
        train_labels, val_labels = all_labels[train_idx], all_labels[val_idx]

        # 构建数据集
        train_dataset = CIFAR10Dataset(train_imgs, train_labels, train_image_dir, train_transform)
        val_dataset = CIFAR10Dataset(val_imgs, val_labels, train_image_dir, test_transform)

        # 训练
        train_loss, val_loss, train_acc, val_acc = train(train_dataset, val_dataset, k=fold+1)

        # 统计
        total_train_loss += train_loss
        total_test_loss += val_loss
        total_train_acc += train_acc
        total_test_acc += val_acc

    print(f'K折交叉验证结果:\n'
          f'平均训练损失: {total_train_loss/k:.4f}, 平均验证损失: {total_test_loss/k:.4f}, '
          f'平均训练准确率: {total_train_acc/k:.4f}, 平均验证准确率: {total_test_acc/k:.4f}')
#------------------------ 用全部数据集训练 ------------------------
def train_full_data():
    # 读取数据
    train_data = pd.read_csv(f"{dataset_dir}/trainLabels.csv")
    le = LabelEncoder()
    train_data['label'] = le.fit_transform(train_data.iloc[:, 1])

    all_imgs = train_data.iloc[:, 0].values
    all_labels = train_data['label'].values

    # 构建Dataset
    full_dataset = CIFAR10Dataset(all_imgs, all_labels, train_image_dir, train_transform)

    # 使用train_transform，因为不再划分验证集
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 初始化模型
    net = ResNet().to(device)

    # 优化器和学习率
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    print("开始用全部数据训练")
    for epoch in range(num_epochs):
        net.train()
        total_loss = 0.0
        correct, total = 0, 0

        for X, y in full_loader:
            X, y = X.to(device), y.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = net(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / len(full_loader)
        acc = correct / total
        scheduler.step()

        print(f'Epoch: {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}')

    # 保存模型
    torch.save(net.state_dict(), 'cifar10_model_full.pth')
    print("模型已保存为 leaf_model_full.pth")

    return net, le

# -----------------------预测-----------------------
# def predict_test(model, label_encoder):
#     test_data = pd.read_csv(f"{image_dir}/test.csv")
#     test_imgs = test_data.iloc[:, 0].values

#     test_dataset = LeafDataset(test_imgs, labels=[0]*len(test_imgs), img_dir=image_dir, transform=test_transform)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

#     model.eval()
#     preds = []

#     with torch.no_grad():
#         for X, _ in test_loader:
#             X = X.to(device)
#             outputs = model(X)
#             pred = outputs.argmax(dim=1).cpu().numpy()
#             preds.extend(pred)

#     # 转换回原始标签
#     preds_labels = label_encoder.inverse_transform(preds)

#     submission = pd.DataFrame({
#         'image': test_imgs,
#         'label': preds_labels
#     })
#     submission.to_csv('submission.csv', index=False)
#     print("提交文件已保存为 submission.csv")
# ---------------------- 8. 运行 ----------------------
if __name__ == "__main__":
    train_k_fold(k)
    # 1️⃣ 用全部数据训练
    #model, label_encoder = train_full_data()

    # 2️⃣ 用训练好的模型做测试集预测并保存提交文件
    #predict_test(model, label_encoder)
