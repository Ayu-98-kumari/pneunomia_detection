# Pneumonia Detection from Chest X-Rays

Binary classification of chest X-ray images into **NORMAL** and **PNEUMONIA** using PyTorch CNNs and transfer learning.

## Dataset

Uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset. Place the extracted data so the directory looks like:

```
Pneumonia_project/
├── chest_xray/
│   └── chest_xray/
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       └── test/
│           ├── NORMAL/
│           └── PNEUMONIA/
└── pneumonia_detection/
    └── (this project)
```

## Setup

```bash
cd pneumonia_detection
pip install -r requirements.txt
```

## Experiments

There are 5 experiments, each defined by a config file:

| # | Config | Architecture | Weighted Sampling | Augmentation | Epochs | LR |
|---|--------|-------------|-------------------|--------------|--------|----|
| 1 | `configs/baseline.yaml` | 3-layer CNN | No | No | 8 | 0.001 |
| 2 | `configs/weighted_sampling.yaml` | 3-layer CNN | Yes | No | 8 | 0.001 |
| 3 | `configs/augmentation.yaml` | 3-layer CNN | Yes | Yes | 15 | 0.001 |
| 4 | `configs/deeper_cnn.yaml` | 5-layer CNN + Dropout | Yes | Yes | 8 | 0.001 |
| 5 | `configs/resnet18.yaml` | Pretrained ResNet-18 | Yes | Yes | 5 | 0.0001 |

## Training

Train a single experiment:

```bash
python train.py --config configs/baseline.yaml
```

Train all experiments one by one:

```bash
python train.py --config configs/baseline.yaml
python train.py --config configs/weighted_sampling.yaml
python train.py --config configs/augmentation.yaml
python train.py --config configs/deeper_cnn.yaml
python train.py --config configs/resnet18.yaml
```

Trained models are saved to the `checkpoints/` directory.

## Evaluation

Evaluate a trained model on the test set:

```bash
python evaluate.py --config configs/baseline.yaml
```

Evaluate all experiments:

```bash
python evaluate.py --config configs/baseline.yaml
python evaluate.py --config configs/weighted_sampling.yaml
python evaluate.py --config configs/augmentation.yaml
python evaluate.py --config configs/deeper_cnn.yaml
python evaluate.py --config configs/resnet18.yaml
```

To evaluate with a specific checkpoint file:

```bash
python evaluate.py --config configs/resnet18.yaml --checkpoint path/to/model.pth
```

Evaluation prints confusion matrix, classification report, and threshold analysis for each config's defined thresholds.

## Project Structure

```
pneumonia_detection/
├── configs/                 # Experiment configs (one per experiment)
│   ├── baseline.yaml
│   ├── weighted_sampling.yaml
│   ├── augmentation.yaml
│   ├── deeper_cnn.yaml
│   └── resnet18.yaml
├── src/                     # Source modules
│   ├── dataset.py           # Data loading, transforms, weighted sampler
│   ├── models.py            # CNN, DeeperCNN, ResNet-18 architectures
│   ├── train.py             # Training and validation loops
│   ├── evaluate.py          # Inference, metrics, threshold analysis
│   └── utils.py             # Config loading, device selection, seeding
├── train.py                 # Training entry point
├── evaluate.py              # Evaluation entry point
├── requirements.txt
└── .gitignore
```

## Adding a New Experiment

1. Create a new YAML file in `configs/` (copy an existing one as a starting point).
2. Adjust the parameters (model, lr, epochs, augmentation, etc.).
3. Run `python train.py --config configs/your_new_config.yaml`.

No code changes needed unless you're adding a new model architecture (in that case, add it to `src/models.py` and register it in `get_model()`).
