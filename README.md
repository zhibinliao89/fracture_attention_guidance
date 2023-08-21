# CNN Attention Guidance for Improved Orthopedics Radiographic Fracture Classification

This repository contains the codebase which was used in our [paper](https://ieeexplore.ieee.org/abstract/document/9718182) ([ArXiv](https://arxiv.org/abs/2203.10690)).
We apologize that we are unable to publish the two fracture datasets described in the paper.
Therefore, this codebase is adapted to the [CUB200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) dataset as a demonstration of the proposed attention guidance method.

## Installation

GPU VRAM requirements ~14GB

If using Conda

```
conda create --prefix .conda python==3.9 -y
conda activate ./.conda
conda install -c "nvidia/label/cuda-11.7.1" cuda -y
```

Install the python packages, please note that this codebase uses [PyTorch's Transforms v2](https://pytorch.org/vision/stable/auto_examples/plot_transforms_v2.html) to transform image and attention mask.  

```
pip3 install -r requirements.txt
```

## Dataset

Please download the `CUB_200_2011.tgz` from the [official site](https://www.vision.caltech.edu/datasets/cub_200_2011/) and place it under `dataset/CUB/CUB_200_2011.tgz` (leave it zipped).

## Usage

The training parameters, transforms, and CUB dataset code are adapted from [zhangyongshun](https://github.com/zhangyongshun/resnet_finetune_cub)'s repository.

Please use the following script to run a vanilla training of a ResNet50 model without attention guidance:

```
cd modelling
source run_original.sh
```
or with attention guidance:

```
cd modelling
source run_attention_guidance.sh
```

## CUB Test Results

The test accuracy (%) on the full test set at the end of the 95th epoch:

| #Training Images per Class        | 5    | 10   | 15   | 20   |
|-----------------------------------|------|------|------|------|
| Original ResNet Model             | 26.0 | 52.2 | 65.2 | 72.9 |
| Attention Guidance (lambda=0.001) | 50.9 | 65.8 | 71.0 | 76.1 |
| Attention Guidance (lambda=0.01)  | 52.0 | 65.6 | 70.8 | 75.3 |
| Attention Guidance (lambda=0.1)   | 50.0 | 65.7 | 71.5 | 75.7 |

* From the above table, it can be seen that the human attention guidance acts as a training regularization and can achieve 
significant improvement over the baseline model when limited training data are available. 
When the number of labelled images goes up, the improvement over the baseline decreases.
This means when a large amount of data is available, the attention should be automatically and accurately formed by the 
image classification model, leaving the effort of adding additional human attention guidance trivial.

## Implementation Notes

### Formula Variation

Please note that this implementation is slightly different from the description in the paper, i.e., in Eq. (4), we do not
upsize the spatial dimensions of the $\mathbf{M}_c + \mathbf{b}_c$ with the `bicubic` interpolation function ($g$) anymore.
This is because that the large number of classes in CUB causes an excessive GPU memory usage if the CAMs are upsized.
Instead, we use the `nearest` interpolation to downsize the spatial dimensions of $\mathbf{S}_c$ in Eq. (6). 
This modification is in [modelling/train_utils.py](modelling/train_utils.py) at line 81.

### CUB Attention Generation

CUB has annotated bird parts marked as 2D coordinates of the center of the parts. For each image, we draw a filled 
circle with `50` pixel radius around each part locations to simulate the attention guidance mask.
The size and shape (if not a circle) of the attention mask will likely to cause difference in test performance.
This process is written in the `_extract_parts()` function in [modelling/dataset.py](modelling/dataset.py) at line 235.

## Supplementary Material

The supplementary material of the paper can be found [here](documentation/supplementary.pdf).