import argparse
import os

import cv2
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
)
from torchvision.transforms import functional as TF
from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):

    def __init__(self, gt_path, output_path):
        super().__init__()
        self.gt_path = gt_path
        self.output_path = output_path
        gt_files = os.listdir(gt_path)
        output_files = os.listdir(output_path)
        # 只保留gt_files和output_files的交集
        self.files = [f for f in gt_files if f in output_files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_name = self.files[index]
        gt_file = os.path.join(self.gt_path, file_name)
        output_file = os.path.join(self.output_path, file_name)
        gt = cv2.cvtColor(cv2.imread(gt_file), cv2.COLOR_BGR2RGB)
        output = cv2.cvtColor(cv2.imread(output_file), cv2.COLOR_BGR2RGB)
        gt = TF.to_tensor(gt)
        output = TF.to_tensor(output)
        return gt, output


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--gt_dir", type=str, required=True)
    argparser.add_argument("--eval_dir", type=str, required=True)
    argparser.add_argument("--device", type=str, default="cuda")

    args = argparser.parse_args()

    device = torch.device(args.device)

    psnr_cal = PeakSignalNoiseRatio(data_range=1).to(device=device)
    ssim_cal = StructuralSimilarityIndexMeasure(data_range=1).to(device=device)
    lpips_cal = LearnedPerceptualImagePatchSimilarity().to(device=device)

    dataset = Dataset(args.gt_dir, args.eval_dir)
    dataloader = DataLoader(dataset)

    mean_psnr = 0
    mean_ssim = 0
    mean_lpips = 0

    for gt, output in tqdm(dataloader):
        gt = gt.to(device=device)
        output = output.to(device=device)
        mean_psnr += psnr_cal(output, gt).item()
        mean_ssim += ssim_cal(output, gt).item()
        mean_lpips += lpips_cal(output, gt).item()

    mean_psnr /= len(dataset)
    mean_ssim /= len(dataset)
    mean_lpips /= len(dataset)
    print(f"PSNR: {mean_psnr:.4f}")
    print(f"SSIM: {mean_ssim:.4f}")
    print(f"LPIPS: {mean_lpips:.4f}")
