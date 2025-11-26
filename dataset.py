
import os
import glob
import json
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# (super class, sub class) -> class_id
# TODO: 실제로 사용할 클래스만 남겨서 정리하면 됨
CLASS_MAP = {
    ("person", "01"): 0,
    ("road_etc", "01"): 1,
    ("road_etc", "05"): 2,
}
NUM_CLASSES = len(CLASS_MAP)


class YoloDataset(Dataset):
    """
    NIPA 어린이 보호구역 프로젝트용 YOLO Dataset

    Args:
        img_dir (str or Path): 이미지 폴더 경로
        label_dir (str or Path): JSON 라벨 폴더 경로
        img_size (int): 리사이즈 기준 한 변 크기 (예: 416)
        transform: torchvision.transforms.Compose 등 (없으면 기본 Transform 사용)
    """

    def __init__(self, img_dir, label_dir, img_size=416, transform=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size

        # transform을 안 넘기면 기본 Resize + ToTensor 사용
        if transform is None:
            self.transform = T.Compose(
                [
                    T.Resize((img_size, img_size)),
                    T.ToTensor(),  # (3, H, W), [0, 1]
                ]
            )
        else:
            self.transform = transform

        # 이미지 파일 목록 (필요 시 확장자 추가)
        self.img_files = sorted(self.img_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.img_files)

    def load_image(self, index: int) -> Image.Image:
        img_path = self.img_files[index]
        img = Image.open(img_path).convert("RGB")
        return img

    def load_labels(self, index: int) -> torch.Tensor:
        """
        JSON 라벨을 읽어서 [class_id, x_center, y_center, w, h] (0~1) 형식으로 반환
        """
        img_path = self.img_files[index]
        base = img_path.stem
        label_path = self.label_dir / f"{base}.json"

        # 라벨 파일이 없으면 빈 박스 반환
        if not label_path.exists():
            return torch.zeros((0, 5), dtype=torch.float32)

        # JSON 읽기
        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_w = data["info"]["width"]
        img_h = data["info"]["height"]

        boxes = []
        for ann in data["annotations"]:
            # 1) 클래스 매핑
            super_cls = ann["object super class"]
            sub_cls = ann["object sub class"]
            key = (super_cls, sub_cls)

            # 정의한 CLASS_MAP에 없는 클래스는 스킵
            if key not in CLASS_MAP:
                continue

            class_id = CLASS_MAP[key]

            # 2) bbox 또는 polygon 처리
            if "bbox" in ann:
                # bbox: [x, y, w, h] (픽셀, 좌상단 기준)
                x, y, w, h = ann["bbox"]
            elif "polygon" in ann:
                # polygon: [x1, y1, x2, y2, ...]
                poly = ann["polygon"]
                xs = poly[0::2]
                ys = poly[1::2]
                x_min, x_max
