import glob
import os
import random
import shutil
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

TRAIN_RATIO, VALID_RATIO, TEST_RATIO = 0.6, 0.2, 0.2


class FileSet(Dataset):

    def __init__(self, path: List[Tuple[str, str]]) -> None:
        super().__init__()
        self.path = path

    def __len__(self):
        return len(self.path)

    def __getitem__(self, index) -> Tuple[str, str]:
        return self.path[index]


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_banding_box(path):
    tree = ET.parse(path)
    boxes = []
    for obj in tree.getroot().findall("object"):
        box = obj.find("bndbox")
        name = obj.find("name")
        if box is None or name is None or name.text == "nonbanded":
            continue
        xmin = int(box.find("xmin").text)  # type: ignore
        ymin = int(box.find("ymin").text)  # type: ignore
        xmax = int(box.find("xmax").text)  # type: ignore
        ymax = int(box.find("ymax").text)  # type: ignore
        boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def is_overlap(box1, box2):
    """
    bos: (xmin, ymin, xmax, ymax)
    """
    return not (
            box1[2] < box2[0] or box1[3] < box2[1] or box2[2] < box1[0] or box2[3] < box1[1]
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--seed", type=int, default=3407)

    args = parser.parse_args()
    seed_everything(args.seed)
    path = args.data_dir
    gt_root = os.path.join(path, "Original Images")
    banded_root = os.path.join(
        path, "Quantized Images with XML files for Banded and Non Banded regions"
    )
    gt_paths = glob.glob(os.path.join(gt_root, "*.png"))
    os.makedirs(os.path.join(args.save_dir, 'gt'))
    os.makedirs(os.path.join(args.save_dir, 'banded'))
    random.shuffle(gt_paths)
    print(f"total image: {len(gt_paths)}")
    trace = dict()
    total_patches = 0
    h_start, w_start = 0, 0
    shape, stride = 256, 75
    for gt_path in tqdm(gt_paths, desc='make patches'):
        basename = os.path.basename(gt_path)
        banded_path = os.path.join(banded_root, basename)
        label_path = os.path.join(banded_root, os.path.splitext(basename)[0] + ".xml")
        if os.path.exists(label_path) and os.path.exists(banded_path):
            gt = cv2.imread(gt_path)
            banded = cv2.imread(banded_path)
            boxes = get_banding_box(label_path)
            h, w, _ = gt.shape
            patches = []
            for y in range(w_start, w - shape + 1, stride):
                for x in range(h_start, h - shape + 1, stride):
                    box = (x, y, x + shape, y + stride)
                    # get banded region only
                    if any(is_overlap(box, b) for b in boxes):
                        patch_gt = gt[x: x + shape, y: y + shape, :]
                        patch_banded = banded[x: x + shape, y: y + shape, :]
                        cv2.imwrite(
                            f"{os.path.join(args.save_dir, 'gt', f'{basename}_{x}_{y}.png')}",
                            patch_gt,
                        )
                        cv2.imwrite(
                            f"{os.path.join(args.save_dir, 'banded', f'{basename}_{x}_{y}.png')}",
                            patch_banded,
                        )
                        patches.append(f"{basename}_{x}_{y}.png")
            total_patches += len(patches)
            trace.update({basename: patches})

    valid_size = int(VALID_RATIO * total_patches)
    test_size = int(TEST_RATIO * total_patches)
    train_size = total_patches - valid_size - test_size

    for p in ["train", "valid", "test"]:
        os.makedirs(os.path.join(args.save_dir, p, "gt"))
        os.makedirs(os.path.join(args.save_dir, p, "banded"))


    def move(src, target):
        for p in src:
            shutil.move(
                os.path.join(args.save_dir, "gt", p), os.path.join(target, "gt")
            )
            shutil.move(
                os.path.join(args.save_dir, "banded", p),
                os.path.join(target, "banded"),
            )


    def copy(src, target):
        shutil.copy(os.path.join(gt_root, src), os.path.join(target, "gt"))
        shutil.copy(os.path.join(banded_root, src), os.path.join(target, "banded"))


    train_cnt, valid_cnt, test_cnt = 0, 0, 0
    train_fhd_cnt, valid_fhd_cnt, test_fhd_cnt = 0, 0, 0
    for basename, patches in tqdm(trace.items(), desc='make train/valid/test set'):
        if train_cnt < train_size:  # use patches for training
            move(patches, os.path.join(args.save_dir, "train"))
            train_cnt += len(patches)
            train_fhd_cnt += 1
        elif valid_cnt < valid_size:  # preserve FHD images for validation
            copy(basename, os.path.join(args.save_dir, "valid"))
            valid_cnt += len(patches)
            valid_fhd_cnt += 1
        else:  # preserve FHD images for testing
            copy(basename, os.path.join(args.save_dir, "test"))
            test_cnt += len(patches)
            test_fhd_cnt += 1

    print(f"{train_cnt}\t{valid_cnt}\t{test_cnt}")
    print(f"{train_fhd_cnt}\t{valid_fhd_cnt}\t{test_fhd_cnt}")
    shutil.rmtree(os.path.join(args.save_dir, 'gt'))
    shutil.rmtree(os.path.join(args.save_dir, 'banded'))
