# DINO Reimplementation

A from-scratch PyTorch reimplementation of **DINO: Emerging Properties in Self-Supervised Vision Transformers** (Caron et al., 2021), trained on the Imagenette dataset. This project implements the full self-distillation pipeline — multi-crop augmentation, a Vision Transformer backbone, student-teacher EMA updates, and the centering + sharpening mechanism that prevents representation collapse — along with downstream evaluation via k-NN classification and linear probing.

## Overview

DINO trains a Vision Transformer with no labels by having a **student** network match the output distribution of a **teacher** network across different augmented views of the same image. The teacher is never trained directly — it's an exponential moving average (EMA) of the student's weights — and its outputs are centered and sharpened to prevent the trivial solution of collapsing to a constant output. The result is a backbone whose learned features transfer well to downstream tasks without ever seeing a label during pretraining.

This repo implements that pipeline end to end:

1. **Self-supervised pretraining** (`Main.py`) — multi-crop DINO training with student/teacher ViTs
3. **Evaluation** (`test.py`) — k-NN and linear-probe accuracy on frozen DINO features, plus the supervised baseline's test accuracy

## Repository Structure

```
DINO-Reimplementation/
├── Main.py                  # DINO self-supervised pretraining entry point
├── test.py                   # k-NN + linear probe evaluation on frozen teacher features
├── DINOLoss.py                # DINO loss: centering, sharpening, cross-entropy over multi-crop pairs
├── DINOHead.py                # Projection head: MLP + L2-normalize + weight-normalized final layer
├── MultiCropWrapper.py        # Groups variable-resolution crops, runs backbone once per resolution
├── ViT/
│   └── ViT.py                 # Vision Transformer backbone
├── DATA/
│   └── Dataset.py             # Imagenette dataset loader (train/val splits)
├── Engine/
│   ├── Train.py                # Per-epoch training loop, CSV logging, EMA teacher update
│   └── Validate.py             # Validation loop
├── Configs/
│   └── *.yaml                  # Experiment configs (model dims, augmentation params, hyperparameters)
└── Saves/
    └── {ExperimentName}/
        ├── {ExperimentName}Train.csv     # epoch, train loss
        ├── Teacher.pt                     # Final EMA teacher checkpoint (used for all downstream eval)
        ├── Student.pt                     # Final student checkpoint 
        └── {ExperimentName}_results.json  # k-NN accuracy, linear probe accuracy
```

## Method

### Multi-crop augmentation
Each training image is augmented into 8 views: **2 global crops** (224×224, covering a large portion of the image) and **6 local crops** (96×96, covering smaller regions). The teacher only ever sees the 2 global crops; the student sees all 8. This asymmetry — "local-to-global correspondence" — is central to what DINO learns: the student is forced to predict the teacher's global-context output from a partial, local view.

Augmentation per crop includes random resized cropping, horizontal flip, color jitter, random grayscale, Gaussian blur, and solarization, with different probabilities per crop type (the two global crops are deliberately augmented asymmetrically — one always blurred, the other rarely blurred but sometimes solarized — so the two global views are genuinely different corruptions of the same image, not near-duplicates).

### Student-teacher architecture
Both student and teacher share the same ViT backbone + projection head architecture (`MultiCropWrapper` + `DINOHead`), but:
- The **student** is trained by backpropagation.
- The **teacher** is never trained directly. After each step, its weights are updated as an EMA of the student's weights: `teacher = momentum * teacher + (1 - momentum) * student`.
- Only the teacher's weights are used for downstream evaluation — it is a smoothed, higher-quality version of the student at any point in training, analogous to Polyak/weight averaging.

### Preventing collapse
Without intervention, this setup has a trivial solution: both networks output a constant, and loss goes to zero without learning anything. DINO prevents this two ways, both implemented in `DINOLoss.py`:
- **Centering** — an EMA-updated per-dimension mean is subtracted from the teacher's output before softmax, preventing any single dimension from dominating.
- **Sharpening** — a lower softmax temperature on the teacher than the student produces a peakier target distribution, encouraging confident predictions without collapsing to uniform outputs.

### Projection head
`DINOHead` maps backbone embeddings through an MLP, L2-normalizes the bottleneck representation, and projects to a high-dimensional output (65536-d by default) through a weight-normalized linear layer — this is the head DINO's loss operates on, and it's discarded after pretraining; only the backbone embedding is used for downstream evaluation.

## Setup

```bash
git clone https://github.com/Nakshu35/DINO-Reimplementation.git
cd DINO-Reimplementation
pip install torch torchvision pyyaml scikit-learn tqdm
```

Download [Imagenette](https://github.com/fastai/imagenette) and point `DatasetDir` in your config YAML to its location.

## Usage

### 1. Pretrain DINO (self-supervised)
```bash
python Main.py --Config Configs/Experiment1.yaml
```
This trains the student/teacher pair with multi-crop augmentation, logs `trainloss`, per step to `Saves/{ExperimentName}/{ExperimentName}Train.csv`, and checkpoints the teacher after every epoch, with a final `Teacher.pt` saved at the end.

### 2. Evaluate
```bash
python test.py --Config Configs/Experiment1.yaml
```
Loads the frozen teacher backbone, extracts embeddings once for the train and validation splits, then runs:
- **k-NN classification** — no training, votes on validation labels using nearest neighbors in embedding space against the train set.
- **Linear probing** — trains a single linear layer on top of the frozen embeddings, then evaluates it on the validation set.

Results are written to `Saves/{ExperimentName}/{ExperimentName}_results.json`:
```json
{
  "experiment": "Exp1",
  "knn_accuracy": 0.0,
  "linear_probe_accuracy": 0.0
}
```

## Config

Example `Configs/Experiment1.yaml` fields:
```yaml
ExperimentName: Exp1
DatasetDir: /path/to/imagenette
RootDir: /path/to/DINO
BatchSize: 32
Epochs: 100
EmbeddingsDim: 256
DINOHiddenLayers: 2048
BottleNeckDim: 256
OutputDim: 65536
Momentum: 0.996
NumClasses: 10
```

## Notes on evaluation

k-NN and linear probing are two separate procedures, not one number derived from the same pass: k-NN requires zero training and votes directly on frozen features, while the linear probe trains a small classifier on top of those same frozen features. Both are computed from backbone embeddings only — never the DINO projection head's output, which exists purely to serve the pretraining loss and is discarded afterward. The supervised ViT baseline is trained independently, end-to-end with labels, and answers a different question (a fully-supervised ceiling) than the self-supervised probes do (the quality of representations learned without labels) — both are reported for reference, but they are not directly comparable in what they measure.

## Status

Model is undertraining, so results will be soon.
