import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights
import os
pretrained_net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
net = nn.Sequential(*list(pretrained_net.children())[:-2]) # 去掉ResNet18最后两层
# X = torch.rand(size=(1,3,320,480)) # 卷积核与输入大小无关，全连接层与输入大小有关
# print(net(X).shape)  # 缩小32倍
num_classes = 21
# 给网络最后加上卷积层和转置卷积层
net.add_module('final_conv',nn.Conv2d(512,num_classes,kernel_size=1))  
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes,num_classes,kernel_size=64,padding=16,stride=32))  # 把图像大小变回去
def bilinear_kernel(in_channels, out_channels, kernel_size):
    # 计算双线性插值核中心点位置
    # 计算双线性插值核的尺寸的一半，由于我们希望中心点位于核的中心，所以需要先计算核的一半大小
    # 我们使用 // 运算符进行整数除法，确保结果为整数
    factor = (kernel_size + 1) // 2
    # 根据核的大小是奇数还是偶数，确定中心点的位置
    # 如果核的大小是奇数，则中心点位于尺寸的一半减去1的位置，因为Python的索引从0开始，所以减去1
    # 例如，如果核的大小是3，那么中心点应该位于1的位置，(3+1)//2 - 1 = 1
    if kernel_size % 2 == 1:
        center = factor - 1
    # 如果核的大小是偶数，则中心点位于尺寸的一半减去0.5的位置
    # 这是因为偶数大小的核没有明确的中心点，所以我们取中间两个元素的平均位置作为中心点
    # 例如，如果核的大小是4，那么中心点应该位于1.5的位置，(4+1)//2 - 0.5 = 1.5
    else:
        center = factor - 0.5
    # 创建一个矩阵，其元素的值等于其与中心点的距离
    og = (torch.arange(kernel_size).reshape(-1,1),
         torch.arange(kernel_size).reshape(1,-1))
    # 计算双线性插值核，其值由中心点出发，向外线性衰减
    filt = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)    
    # 初始化一个权重矩阵，大小为 (输入通道数, 输出通道数, 核大小, 核大小)
    weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))  
    # 将双线性插值核的值赋给对应位置的权重
    weight[range(in_channels),range(out_channels),:,:] = filt
    # 返回初始化的权重矩阵，这个权重矩阵可以直接用于初始化转置卷积层的权重
    return weight

# 初始化转置卷积层的权重
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)
batch_size, crop_size = 32, (320,480)


# 数据集相关
VOC_COLORMAP = [[0,0,0],[128,0,0],[0,128,0],[128,128,0],
               [0,0,128],[128,0,128],[0,128,128],[128,128,128],
               [64,0,0],[192,0,0],[64,128,0],[192,128,0],
               [64,0,128],[192,0,128],[64,128,128],[192,128,128],
               [0,64,0],[128,64,0],[0,192,0],[128,192,0],
               [0,64,128]]
# 定义函数voc_colormap2label，这个函数用于建立一个映射，将RGB颜色值映射为对应的类别索引
def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射"""
    # 创建一个全零的张量，大小为256的三次方，因为RGB颜色的每个通道有256种可能的值，所以总共有256^3种可能的颜色组合。数据类型设为long
    colormap2label = torch.zeros(256**3, dtype=torch.long)
    # 对于VOC_COLORMAP中的每个颜色值（colormap）
    for i, colormap in enumerate(VOC_COLORMAP):
        # 计算颜色值的一维索引，并将这个索引对应的位置设为i。这样，给定一个颜色值，我们就可以通过这个映射找到对应的类别索引
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i  
    # 返回映射
    return colormap2label

# 定义函数voc_label_indices，这个函数用于将一张标签图像中的每个像素的颜色值映射为对应的类别索引
def voc_label_indices(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引"""
    # 将输入的colormap的通道维度移到最后一维，并将其转换为numpy数组，然后转换为int32类型。这是因为我们需要使用numpy的高级索引功能
    colormap = colormap.permute(1,2,0).numpy().astype('int32')
    # 计算colormap中每个像素的颜色值对应的一维索引。这里的索引计算方式和上一个函数中的是一致的
    idx = ((colormap[:,:,0] * 256 + colormap[:,:,1]) * 256 + colormap[:,:,2])  
    # 使用colormap2label这个映射将索引映射为对应的类别索引，并返回
    return colormap2label[idx]
# 自定义语义分割数据集类
# 定义一个自定义的数据集类，用于加载VOC数据集。这个类继承了torch.utils.data.Dataset
class VOCSegDataset(torch.utils.data.Dataset):
    """一个用于加载VOC数据集的自定义数据集"""
    # 初始化方法，输入参数：is_train表示是否是训练集，crop_size是裁剪后的图像尺寸，voc_dir是VOC数据集的路径
    def __init__(self, is_train, crop_size, voc_dir):
        # 定义一个归一化变换，用于对输入图像进行归一化。这里使用了ImageNet数据集的均值和标准差
        self.transform = torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])    
        # 保存裁剪尺寸到self.crop_size
        self.crop_size = crop_size
        # 调用之前定义的read_voc_images函数，读取VOC数据集的特征图像和标签图像
        features, labels = read_voc_images(voc_dir, is_train = is_train)
        # 对读取到的特征图像进行筛选和归一化处理，并保存到self.features中
        self.features = [self.normalize_image(feature) for feature in self.filter(features)]  
        # 对读取到的标签图像进行筛选处理，并保存到self.labels中
        self.labels = self.filter(labels)
        # 创建一个从颜色映射到类别索引的映射表，并保存到self.colormap2label中
        self.colormap2label = voc_colormap2label()
        # 打印出读取到的图像数量
        print('read ' + str(len(self.features)) + ' examples')
        
    # 定义一个方法，用于对输入图像进行归一化处理
    def normalize_image(self, img):
        # 将输入图像转换为浮点数类型，并调用归一化变换对其进行处理，最后返回处理后的图像
        return self.transform(img.float())
    
    # 定义一个方法，用于筛选符合要求的图像。筛选条件是图像的高度和宽度都大于等于裁剪尺寸
    def filter(self, imgs):
        # 返回筛选后的图像列表
        return [img 
                for img in imgs 
                if (img.shape[1] >= self.crop_size[0] and img.shape[2] >= self.crop_size[1] ) ]
    
    # 每一次返回的样本做一次rand_crop
    # 定义一个方法，用于根据索引获取一个样本。这是一个实现Dataset接口所必须的方法
    def __getitem__(self, idx):
        # 调用之前定义的voc_rand_crop函数，对指定索引的特征图像和标签图像进行随机裁剪
        feature, label = voc_rand_crop(self.features[idx],self.labels[idx],*self.crop_size)   
        # 调用voc_label_indices函数，将裁剪后的标签图像转换为类别索引，并返回裁剪和转换后的特征图像和标签图像
        return (feature, voc_label_indices(label,self.colormap2label))
    # 定义一个方法，用于获取数据集的长度。这是一个实现Dataset接口所必须的方法
    def __len__(self):
        # 返回特征图像列表的长度，即数据集的长度
        return len(self.features)
    
def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像并标注"""
    # 定义txt_fname，这是我们将读取的文件的路径。如果is_train为True，我们读取的是'train.txt'，否则是'val.txt'。这些文件中包含了我们需要读取的图像的文件名
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                            'train.txt' if is_train else 'val.txt')
    # 设定图像的读取模式为RGB
    mode = torchvision.io.image.ImageReadMode.RGB
    # 读取txt_fname文件，并将文件中的内容分割成一个个的文件名，然后存储在images列表中   
    with open(txt_fname,'r') as f:
        images = f.read().split()
    # 创建两个空列表，分别用于存储特征和标签
    features, labels = [],[]
    # 对于images中的每个文件名
    for i, fname in enumerate(images):
        # 使用torchvision.io.read_image读取对应的图像文件，然后添加到features列表中
        features.append(torchvision.io.read_image(os.path.join(voc_dir,'JPEGImages',f'{fname}.jpg')))  
        # 使用torchvision.io.read_image读取对应的标签图像文件，然后添加到labels列表中
        labels.append(torchvision.io.read_image(os.path.join(voc_dir,'SegmentationClass',f'{fname}.png'),mode))  
    # 返回特征和标签列表
    return features, labels
def voc_rand_crop(feature, label, height, width):
    """随即裁剪特征和标签图像"""
    # 调用RandomCrop的get_params方法，随机生成一个裁剪框。裁剪框的大小是(height, width)
    # rect拿到特征的框
    rect = torchvision.transforms.RandomCrop.get_params(feature,(height,width))   # 生成一个裁剪框
    # 根据生成的裁剪框，对特征图像进行裁剪
    # 拿到框中的特征和标号
    feature = torchvision.transforms.functional.crop(feature, *rect)
    # 根据生成的裁剪框，对标签图像进行裁剪。注意，我们是在同一个裁剪框下裁剪特征图像和标签图像，以保证它们对应的位置仍然是对齐的
    label = torchvision.transforms.functional.crop(label,*rect)
    # 返回裁剪后的特征图像和标签图像
    return feature, label
# 整合所有组件
# 定义一个函数来加载VOC语义分割数据集，输入参数是批次大小和裁剪尺寸
def load_data_voc(batch_size, crop_size):
    """加载VOC语义分割数据集"""
    # 下载并提取VOC2012数据集，获取数据集的路径
    voc_dir = '/home/kongdechang/python/CV/01_data/VOCdevkit/VOC2012'  
    # 获取数据加载器的工作进程数量
    num_workers = 4
    # 创建一个训练数据加载器实例，输入参数分别为：VOCSegDataset是待加载的数据集，batch_size是批次大小，shuffle=True表示在每个迭代周期中随机打乱数据，drop_last=True表示如果最后一个批次的样本数量小于batch_size，则丢弃该批次，num_workers是数据加载器的工作进程数量
    train_iter = torch.utils.data.DataLoader(VOCSegDataset(True, crop_size, voc_dir), batch_size,shuffle=True,
                                            drop_last = True, num_workers=num_workers)
    # 创建一个测试数据加载器实例，输入参数分别为：VOCSegDataset是待加载的数据集，batch_size是批次大小，drop_last=True表示如果最后一个批次的样本数量小于batch_size，则丢弃该批次，num_workers是数据加载器的工作进程数量
    test_iter = torch.utils.data.DataLoader(VOCSegDataset(False, crop_size, voc_dir), batch_size, drop_last=True,
                                           num_workers=num_workers)
    # 返回训练数据加载器和测试数据加载器
    return train_iter, test_iter


batch_size, crop_size = 32, (320,480)
# 使用 d2l.load_data_voc 函数读取VOC2012数据集
# 此函数将会返回训练和测试的数据迭代器，数据迭代器可以按批次产生数据，方便训练模型
# 在读取数据的过程中，图像会被裁剪到设定的尺寸，并且会进行一些常规的数据增强操作，如随机裁剪、随机翻转等
train_iter, test_iter = load_data_voc(batch_size, crop_size)