import glob
import os
import random
from typing import List, Tuple

import cv2
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.transforms import ToTensor


class DebandDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        hp: float = 0.5,
        vp: float = 0.5,
        min_brightness: float = 0.6,
        max_brightness: float = 1.6,
        data_aug: bool = True,
        phase: str = "train",
    ):
        super(DebandDataset, self).__init__()
        self.phase = phase
        self.data_aug = data_aug
        self.data_path = self._glob_file(data_path)
        self.to_tensor = ToTensor()
        self.brightness = [min_brightness, max_brightness]
        self.hp = hp
        self.vp = vp

    def _fetch_one(self, data_path: Tuple[str, str]) -> Tuple[Tensor, Tensor]:
        clear = cv2.cvtColor(cv2.imread(data_path[0]), cv2.COLOR_BGR2RGB)
        band = cv2.cvtColor(cv2.imread(data_path[1]), cv2.COLOR_BGR2RGB)
        return self.to_tensor(clear), self.to_tensor(band)

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:  # type: ignore
            # print("?")
        clear, band = self._fetch_one(self.data_path[idx])
        if self.data_aug and self.phase == "train":
            # horizontal flip
            is_hflip = torch.rand(1) < self.hp
            # vertical flip
            is_vflip = torch.rand(1) < self.vp
            # rotate angle
            angle = random.choice([0, 90, 180, 270])
            clear, band = self._apply_transform(
                clear,
                band,
                lambda img, is_flip: TF.hflip(img) if is_flip else img,
                is_hflip,
            )
            clear, band = self._apply_transform(
                clear,
                band,
                lambda img, is_flip: TF.vflip(img) if is_flip else img,
                is_vflip,
            )
            clear, band = self._apply_transform(clear, band, TF.rotate, angle)
            # clear, band = self._apply_transform(clear, band, TF.adjust_brightness, lighter_factor)
        return clear, band

    def _apply_transform(self, clear, band, op, *params):
        return op(clear, *params), op(band, *params)

    def _glob_file(self, data_path: str) -> List[Tuple[str, str]]:
        banded_paths = glob.glob(os.path.join(data_path, "banded", "*.png"))
        gt_root = os.path.join(data_path, "gt")
        data_paths = []
        for banded_path in banded_paths:
            gt_path = os.path.join(gt_root, os.path.basename(banded_path))
            if os.path.exists(gt_path):
                data_paths.append((gt_path, banded_path))
        for gt_path, banded_path in data_paths:
            assert os.path.basename(gt_path) == os.path.basename(banded_path)
        return data_paths