import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------- 1. 参数设置 ----------------------
k = 5
num_epochs = 70
lr = 0.0001
weight_decay = 0.0001
batch_size = 32
train_image_dir = "./01_data/05_DataSet_ImageNet_Dog/train"
test_image_dir = "./01_data/05_DataSet_ImageNet_Dog/test"
dataset_dir="./01_data/05_DataSet_ImageNet_Dog"

# ---------------------- 2. 自定义 Dataset ----------------------
class CIFAR10Dataset(Dataset):
    def __init__(self, img_names, labels, img_dir, transform=None):
        self.img_names = [str(x) + ".jpg" for x in img_names]
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
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 120)
        )
        # 冻结除fc层以外的参数
        # for name, param in self.resnet.named_parameters():
        #     if name.startswith('layer3') or name.startswith('layer4') or name.startswith('fc'):
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

    def forward(self, x):
        return self.resnet(x)
class ConvNextNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, 120)  # 输出类别

    def forward(self, x):
        return self.backbone(x)
# ---------------------- 4. 预处理函数 ----------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224,scale=(0.08,1.0),ratio=(3.0/4.0, 4.0/3.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                                    [0.229,0.224,0.225]),
    transforms.RandomRotation(15),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.RandomGrayscale(p=0.1),
                                    ])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                                     [0.229,0.224,0.225])])
# ---------------------- 5. 训练函数 ----------------------
def train(train_dataset, test_dataset, k=-1):
    net = ConvNextNet().to(device)
    #optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("开始训练")
    for epoch in range(num_epochs):
        net.train()
        train_loss = 0.0
        correct, total = 0, 0  # 准确率统计

        # 用 tqdm 包裹训练迭代器，显示进度条
        train_bar = tqdm(train_loader, desc=f'K:{k} Epoch:{epoch+1} Train', leave=False)
        for X, y in train_bar:
            X, y = X.to(device), y.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = net(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            train_bar.set_postfix(loss=loss.item(), acc=correct/total)

        train_loss /= len(train_loader)
        train_acc = correct / total
        scheduler.step()

        # 验证阶段
        net.eval()
        test_loss = 0.0
        correct, total = 0, 0

        test_bar = tqdm(test_loader, desc=f'K:{k} Epoch:{epoch+1} Test ', leave=False)
        with torch.no_grad():
            for X, y in test_bar:
                X, y = X.to(device), y.to(device, dtype=torch.long)
                outputs = net(X)
                loss = loss_fn(outputs, y)
                test_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

                test_bar.set_postfix(loss=loss.item(), acc=correct/total)

        test_loss /= len(test_loader)
        test_acc = correct / total

        print(f'K: {k}, Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Acc:{train_acc:.4f}, Test Acc: {test_acc:.4f}')

    return train_loss, test_loss, train_acc, test_acc
# ---------------------- 6. K折交叉验证 ----------------------
from sklearn.model_selection import StratifiedKFold

def train_k_fold(k):
    train_data = pd.read_csv(f"{dataset_dir}/labels.csv")
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
if __name__ == "__main__":
    train_k_fold(k)