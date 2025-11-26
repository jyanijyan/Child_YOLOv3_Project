
## Dataset + collate_fn 뼈대

import os
import glob
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


class YoloDataset(Dataset):
    """
    - img_dir: 이미지 폴더 경로 (예: data/images)
    - label_dir: 라벨 폴더 경로 (예: data/labels)
    - img_size: (정사각형 가정) 예: 416
    - transform: torchvision.transforms.Compose(...) 같은 것 (옵션)
    """
    def __init__(self, img_dir, label_dir, img_size=416, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform

        # jpg, png 등 확장자에 맞게 수정 가능
        self.img_files = sorted(
            glob.glob(os.path.join(img_dir, "*.jpg"))
        )

    def __len__(self):
        return len(self.img_files)

    def load_image(self, index):
        img_path = self.img_files[index]
        img = Image.open(img_path).convert("RGB")
        # resize to (img_size, img_size)
        img = img.resize((self.img_size, self.img_size))
        return img

    def load_labels(self, index):
        img_path = self.img_files[index]
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, base + ".txt")

        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    cls, x, y, w, h = line.split()
                    boxes.append([
                        int(cls),
                        float(x), float(y), float(w), float(h)
                    ])
        if len(boxes) == 0:
            # 박스가 없으면 (0,5) 텐서 반환
            return torch.zeros((0, 5), dtype=torch.float32)
        return torch.tensor(boxes, dtype=torch.float32)

    def __getitem__(self, index):
        img = self.load_image(index)
        targets = self.load_labels(index)

        if self.transform is not None:
            img = self.transform(img)
        else:
            # 기본: [0,1] 정규화 + Tensor 변환
            img = torch.from_numpy(
                (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
                 .view(img.size[1], img.size[0], 3)
                 .numpy().astype("float32") / 255.0)
            ).permute(2, 0, 1)  # (H,W,3) -> (3,H,W)

        # targets: (N, 5) [class, x, y, w, h] (0~1)
        return img, targets



## 배치에서 박스 개수가 달라지므로 collate_fn 을 따로 정의

def yolo_collate_fn(batch):
    """
    batch: list of (image, targets)
      - image: Tensor (3,H,W)
      - targets: Tensor (N_i, 5) [class, x, y, w, h]

    return:
      - images: Tensor (B,3,H,W)
      - targets: Tensor (M,6) [batch_idx, class, x, y, w, h]
    """
    images = []
    targets_list = []

    for i, (img, targets) in enumerate(batch):
        images.append(img)
        if targets.numel() > 0:
            # 각 target에 batch index 붙이기
            batch_idx = torch.full(
                (targets.size(0), 1),
                i,
                dtype=targets.dtype
            )
            targets_with_idx = torch.cat([batch_idx, targets], dim=1)
            targets_list.append(targets_with_idx)

    images = torch.stack(images, dim=0)

    if len(targets_list) > 0:
        targets_all = torch.cat(targets_list, dim=0)
    else:
        targets_all = torch.zeros((0, 6))

    return images, targets_all


## Data Loader에 적용

train_dataset = YoloDataset(
    img_dir="C:\Users\Jian Park\Desktop\JianPark\Google Study Jam\Sample Data\Sample\raw data",
    label_dir="C:\Users\Jian Park\Desktop\JianPark\Google Study Jam\Sample Data\Sample\labeling data",
    img_size=nnn,
    transform=None  # 나중에 Albumentations 등으로 교체 가능하다고 함
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=yolo_collate_fn,
    num_workers=1
)
