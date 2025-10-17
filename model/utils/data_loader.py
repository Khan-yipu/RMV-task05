import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from config.config import config


class ArmorDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []

        self._load_data()

    def _load_data(self):
        """加载数据"""
        for label in range(1, config.NUM_CLASSES + 1):
            class_dir = os.path.join(self.data_dir, str(label))
            if not os.path.exists(class_dir):
                print(f"警告: 目录不存在 {class_dir}")
                continue

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(label - 1)  # 转换为0-4

        print(f"从 {self.data_dir} 加载了 {len(self.images)} 张图片")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"错误加载图片 {img_path}: {e}")
            # 返回黑色图像作为替代
            image = Image.new(
                "RGB", (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), color="black"
            )

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms():
    """获取数据预处理变换 - 针对20×28小尺寸图像优化"""
    train_transform = transforms.Compose(
        [
            transforms.Resize(config.IMAGE_SIZE),  # 确保尺寸统一
            transforms.RandomRotation(5),  # 减小旋转角度
            transforms.RandomHorizontalFlip(0.2),  # 减小翻转概率
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1
            ),  # 减小颜色扰动
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform


def get_data_loaders():
    """获取数据加载器"""
    train_transform, val_transform = get_transforms()

    # 创建完整数据集
    full_dataset = ArmorDataset(config.TRAIN_DIR, transform=train_transform)

    if len(full_dataset) == 0:
        raise ValueError(f"在 {config.TRAIN_DIR} 中没有找到训练数据")

    # 分割训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = train_test_split(
        list(range(len(full_dataset))),
        train_size=train_size,
        test_size=val_size,
        random_state=42,
        stratify=full_dataset.labels,
    )

    # 创建子数据集
    from torch.utils.data import Subset

    train_subset = Subset(full_dataset, train_dataset)
    val_subset = Subset(full_dataset, val_dataset)

    # 为验证集应用不同的transform
    val_subset.dataset.transform = val_transform

    # 创建数据加载器
    train_loader = DataLoader(
        train_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,  # 小图像减少workers
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"训练集: {len(train_subset)} 张图片")
    print(f"验证集: {len(val_subset)} 张图片")

    return train_loader, val_loader


def get_test_loader():
    """获取测试数据加载器"""
    _, test_transform = get_transforms()

    test_dataset = ArmorDataset(config.TEST_DIR, transform=test_transform)

    if len(test_dataset) == 0:
        raise ValueError(f"在 {config.TEST_DIR} 中没有找到测试数据")

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"测试集: {len(test_dataset)} 张图片")

    return test_loader
