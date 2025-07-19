from torch.utils.data import Dataset
from PIL import Image
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import json
class CocoDetectionFRCNNDataset(Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.root = root  # 图像路径
        self.coco = COCO(annFile)
        print(max(self.coco.getCatIds()) + 1)
        self.ids = list(sorted(self.coco.imgs.keys()))
        #self.ids = self.ids[:400]  # 只取前10张图片调试

        self.transforms = transforms
        # 提取所有出现的类别ID
        cat_ids = self.coco.getCatIds()
        self.cat_id_to_label = {cat_id: idx + 1 for idx, cat_id in enumerate(cat_ids)}  # 从 1 开始映射

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]

        path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(path).convert("RGB")

        # 组织目标
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for obj in anns:
            bbox = obj['bbox']
            # COCO 是 [x, y, w, h]，Faster R-CNN 需要 [x1, y1, x2, y2]
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = x1 + bbox[2]
            y2 = y1 + bbox[3]
            boxes.append([x1, y1, x2, y2])
            labels.append(obj['category_id'])  
            areas.append(obj['area'])
            iscrowd.append(obj.get('iscrowd', 0))

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': img_id,
            'area': torch.tensor(areas),
            'iscrowd': torch.tensor(iscrowd)
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)


def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT 
    model = fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model

def mAP():
    # 路径配置
    ann_file = '01_data/06_DataSet_CowBoy/cowboyoutfits/train.json'
    res_file = "/home/kongdechang/python/CV/01_data/06_DataSet_CowBoy/cowboyoutfits/detections.json"  # 你前面保存的模型预测结果

    # 加载 ground truth 和 预测结果
    cocoGt = COCO(ann_file)
    cocoDt = cocoGt.loadRes(res_file)

    # 创建 COCOeval 实例
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')

    # 可选：评估指定 image_id 范围（如果你只用了一部分数据）
    # cocoEval.params.imgIds = cocoGt.getImgIds()

    # 执行评估流程
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
# 定义配置
batch_size=3
epoch_num=5
num_classes=1035

def train_full_data():
     # 参数
    image_root = '01_data/06_DataSet_CowBoy/cowboyoutfits/images'
    ann_path = '01_data/06_DataSet_CowBoy/cowboyoutfits/train.json'

    # Dataset & Dataloader
    dataset = CocoDetectionFRCNNDataset(
        root=image_root,
        annFile=ann_path,
        transforms=T.ToTensor()
    )
    data_loader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes).to(device)

    # 优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # 简化训练循环
    for epoch in range(epoch_num):
        #--- 训练 ---
        model.train()
        epoch_loss = 0.0
        train_bar = tqdm(data_loader, desc=f"Epoch {epoch}", unit="batch")
        for images, targets in train_bar:
            # 用list是因为Faster R-CNN支持图像大小不一致，这样就不能放到一个torch里
            images = list(img.to(device) for img in images)
            targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets) # 各种损失：包括分类损失、回归损失等
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            loss_value = losses.item()
            epoch_loss += loss_value
            train_bar.set_postfix(loss=loss_value)
        print(f"train:[Epoch {epoch}] Average Loss: {epoch_loss / len(data_loader):.4f}")
        # --- 测试 ---
        print("开始测试...")
        model.eval()
        results = []
        test_bar = tqdm(data_loader, desc=f"Epoch {epoch}", unit="batch")
        annFile= '01_data/06_DataSet_CowBoy/cowboyoutfits/train.json'
        coco = COCO(annFile)
        with torch.no_grad():
            for images, targets in test_bar:
                images = [img.to(device) for img in images]
                outputs = model(images)

                for img, output in zip(targets, outputs):  # 用 targets 拿 image_id
                    image_id = img["image_id"]
                    #print(f"预测 image_id: {image_id}, 合法: {image_id in coco.getImgIds()}")
                    image_id = img["image_id"]
                    boxes = output["boxes"].cpu().numpy()
                    scores = output["scores"].cpu().numpy()
                    labels = output["labels"].cpu().numpy()
                    #print("Num predictions:", len(output["boxes"]))

                    for box, score, label in zip(boxes, scores, labels):
                        x1, y1, x2, y2 = box
                        result = {
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(score)
                        }

                        results.append(result)
                        #print(results)
            #test_bar.set_postfix(loss=loss_value)

        with open("/home/kongdechang/python/CV/01_data/06_DataSet_CowBoy/cowboyoutfits/detections.json", "w") as f:
            json.dump(results, f)
        mAP()

            
    # 训练完成后保存模型
    torch.save(model.state_dict(), "fasterrcnn_cowboy.pth")
    print("模型已保存为 fasterrcnn_cowboy.pth")
import pandas as pd
from torchvision import transforms

def create_submission_from_csv(csv_path, image_root, model, output_json_path,
                                categories=None, score_thresh=0.1, device='cuda'):
    """
    读取 valid.csv，调用模型推理，输出COCO格式的detections.json文件
    """
    df = pd.read_csv(csv_path, header=None, names=['id', 'file_name'])
    transform = transforms.ToTensor()
    model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating detections"):
            img_id = int(row['id'])
            file_path = os.path.join(image_root, row['file_name'])

            img = Image.open(file_path).convert("RGB")
            width, height = img.size
            input_tensor = transform(img).unsqueeze(0).to(device)

            output = model(input_tensor)[0]  # 单张图片的输出

            for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
                if score < score_thresh:
                    continue
                x1, y1, x2, y2 = box.tolist()
                pred = {
                    "image_id": img_id,
                    "category_id": int(label if categories is None else categories[label]),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score)
                }
                results.append(pred)

    with open(output_json_path, 'w') as f:
        json.dump(results, f)
    print(f"[✔] 保存成功：{output_json_path}")


if __name__ == "__main__":
    #train_full_data()
    # 模型和权重加载
    model = get_model(num_classes)
    model.load_state_dict(torch.load("fasterrcnn_cowboy.pth"))

    # 调用函数生成 JSON
    create_submission_from_csv(
        csv_path="/home/kongdechang/python/CV/01_data/06_DataSet_CowBoy/cowboyoutfits/valid.csv",
        image_root="01_data/06_DataSet_CowBoy/cowboyoutfits/images",
        model=model,
        output_json_path="/home/kongdechang/python/CV/01_data/06_DataSet_CowBoy/cowboyoutfits/submit.json",
        device='cuda'
    )
