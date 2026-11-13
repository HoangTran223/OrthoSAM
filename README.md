# OrthoSAM: Orthogonal Perturbations for Sharpness-Aware Minimization


## Requirements

The main training entrypoint uses CUDA/NCCL.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
pip install -r requirements.txt
```

## Data

CIFAR-10 and CIFAR-100 are downloaded automatically by torchvision to `dataDir`
from the selected config. The default is `~/.datasets`.

Tiny-ImageNet:

```bash
python scripts/download_tiny_imagenet_hf.py --out ~/.datasets/tiny-imagenet
```

For ImageNet or a shared dataset directory:

```bash
export SAM_DATASET_ROOT=/path/to/datasets
```

## Run

Image-classification settings: 200 epochs; SGD-style momentum `0.9`;
cosine LR after warmup; CIFAR uses random crop, horizontal flip, Cutout
probability `0.5`, and label smoothing `0.1`; Tiny-ImageNet uses RandAugment
magnitude `10` and label smoothing `0.1`.

OrthoSAM is `--optimizer OrthoSAM`, with its perturbation radius set by
`--orthosamRho`. Results are reported as mean and standard deviation over three
independent runs; add `--rndSeed True` when repeating runs for statistics.

### CIFAR-10 / ResNet-50

Config: [configs/CIFAR10_ResNet50.ini](configs/CIFAR10_ResNet50.ini)

Setting: `dataset=CIFAR10`, `model=resnet`, `depth=50`, `batchSize=256`,
`learningRate=0.1`, `weightDecay=0.001`, `momentum=0.9`, `rho=0.05`.

```bash
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.run --standalone --nproc_per_node=1 main.py \
  --ifile configs/CIFAR10_ResNet50.ini \
  --optimizer OrthoSAM \
  --orthosamRho 0.05 \
  --epochs 200
```

Run all image baselines plus OrthoSAM on this config:

```bash
GPU_ID=0 bash scripts/run_sam_variants.sh \
  configs/CIFAR10_ResNet50.ini \
  cifar10_resnet50 \
  --epochs 200
```

### CIFAR-100 / ResNet-50

Config: [configs/CIFAR100_ResNet50.ini](configs/CIFAR100_ResNet50.ini)

Setting: `dataset=CIFAR100`, `model=resnet`, `depth=50`, `batchSize=256`,
`learningRate=0.1`, `weightDecay=0.001`, `momentum=0.9`, `rho=0.045`.

```bash
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.run --standalone --nproc_per_node=1 main.py \
  --ifile configs/CIFAR100_ResNet50.ini \
  --optimizer OrthoSAM \
  --orthosamRho 0.045 \
  --epochs 200
```

Run all image baselines plus OrthoSAM on this config:

```bash
GPU_ID=0 bash scripts/run_sam_variants.sh \
  configs/CIFAR100_ResNet50.ini \
  cifar100_resnet50 \
  --epochs 200
```

### Tiny-ImageNet / ResNet-50

Prepare Tiny-ImageNet first:

```bash
python scripts/download_tiny_imagenet_hf.py --out ~/.datasets/tiny-imagenet
```

Config: [configs/TinyImageNet_ResNet50.ini](configs/TinyImageNet_ResNet50.ini)

Setting: `dataset=TinyImageNet`, `model=resnet`, `depth=50`, `imageSize=64`,
`batchSize=125`, `learningRate=0.05`, `weightDecay=0.001`, `momentum=0.9`,
`randAugment=True`, `randAugment_magnitude=10`, `rho=0.05`.

```bash
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.run --standalone --nproc_per_node=1 main.py \
  --ifile configs/TinyImageNet_ResNet50.ini \
  --optimizer OrthoSAM \
  --orthosamRho 0.05 \
  --epochs 200
```

Run all image baselines plus OrthoSAM on this config:

```bash
GPU_ID=0 bash scripts/run_sam_variants.sh \
  configs/TinyImageNet_ResNet50.ini \
  tinyimagenet_resnet50 \
  --epochs 200
```

Tiny-ImageNet / ResNet-101 uses
[configs/TinyImageNet_ResNet101.ini](configs/TinyImageNet_ResNet101.ini):

```bash
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.run --standalone --nproc_per_node=1 main.py \
  --ifile configs/TinyImageNet_ResNet101.ini \
  --optimizer OrthoSAM \
  --orthosamRho 0.05 \
  --epochs 200
```

Logs are written under `logs/`. Relative `logDir = logs` entries in
[configs](configs) are resolved from the repository root.

## Other Presets

Config files are in [configs](configs). Supported datasets are `CIFAR10`,
`CIFAR100`, and `TinyImageNet`.

Supported model names are `WRN`, `resnet`, `resnetCIFAR`, `VGG`, `PyramidNet`,
`DenseNet`, and `ViT`.
