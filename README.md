# Gradient-Guided Dual-Branch Network for Efficient Image Debanding


This repository contains the PyTorch implementation of the paper "Gradient-Guided Dual-Branch Network for Efficient Image Debanding".


## Abstract

Banding artifacts, a common form of image distortion, degrade image quality due to factors such as bit-depth limitations, image compression, and excessive editing. To address this issue, we propose a gradient-guided dual-branch densely self-guided network specifically designed for image debanding. Our network employs a top-down self-guidance architecture to capture multi-scale and contextual information, crucial for effective image debanding. The dual-branch, comprising a detail branch and a gradient branch, operates at full-resolution level, tasked with generating non-banded images and the corresponding gradient maps, respectively. To facilitate gradient guidance from the gradient branch to the detail branch, we design a feature transmission module and a fusion block, compelling the detail branch to focus on image debanding and detail preservation. Experimental results demonstrate that our proposed network achieves superior performance in both quantitative metrics and visual quality, outperforming existing debanding methods.


## Example

### Banded Image

![banded_image](./example/banded.png)

### Debanded Image with our method

![deband_image](./example/debanding_result.png)

## Installation

```
conda create -n GGDBN python=3.10
conda activate GGDBN
cd path/to/GGDBN
pip install -r requirements.txt
```

## Data Preparation

We prepare the dataset follow the prior work [Zhou et al. (2022)](https://github.com/RaymondLZhou/deepDeband), generating data from [Kapoor et. al. (2021)](https://github.com/akshay-kap/Meng-699-Image-Banding-detection).

1. Download the dataset from the following links: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4513740.svg)](https://doi.org/10.5281/zenodo.4513740).

2. Extract the dataset and run the data_prepare.py as follows:

```
python data_prepare.py \
    --data_path path/to/dataset \
    --save_path path/to/save_path
```

## Train

You can modify the script/train.sh to fit your own settings and then run the following command:

```
bash script/train.sh
```

### Pretrained Model

You can download the pretrained model from [Google Drive](https://drive.google.com/drive/folders/1WJ8P81H0a2aFY_RnuvLYUbv0lZU1xLoO?usp=sharing).


## Test

You can modify the script/test.sh to run the model on your own images and then run the following command:

```
bash script/test.sh
```

