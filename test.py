import argparse
import os
import time

import cv2
import torch
from torch import nn
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from tqdm import tqdm

from model import SGN_with_Guide, GGDBN

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="The input banded image directory")
    parser.add_argument("--ckpt_path", type=str, help="The pth checkpoint file path")

    parser.add_argument(
        "--output_dir", type=str, help="The output directory, default is data_dir/output"
    )

    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")

    args = parser.parse_args()
    path = args.data_dir
    if args.output_dir is None:
        args.output_dir = os.path.join(path, "output")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    device = torch.device(args.device)

    raw = SGN_with_Guide(3, 3, 32, act_op=nn.PReLU).to(device)
    model = GGDBN(raw, device=device).to(device)
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.no_grad():
        start = time.time()
        for img_path in tqdm(os.listdir(path)):
            if not img_path.endswith(('.jpg', '.png', '.jpeg')):
                continue
            img_path = os.path.join(path, img_path)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = TF.to_tensor(img).unsqueeze(0).to(device)
            out, grad = model(img)
            # out = out.clamp(0, 1)
            save_image(out, os.path.join(args.output_dir, os.path.basename(img_path)))

        times = time.time() - start
    print(f"avg time: {times / len(os.listdir(path)):.4f}s")
