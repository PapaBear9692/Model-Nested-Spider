# Model-Nested-Spider

An optimized upgrade to the original [Model-Spider](https://github.com/zhangyikaii/Model-Spider.git) framework for **faster and efficient pre-trained model (PTM) selection**. This implementation features improved architecture and hierarchical model organization for accelerated best model ranking.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How to Run](#how-to-run)
- [Understanding the Program Flow](#understanding-the-program-flow)
- [Output & Results](#output--results)
- [Baseline Methods](#baseline-methods)
- [References](#references)

---

## Overview

**Model-Nested-Spider** learns to rank pre-trained models by their suitability for downstream tasks. Given a new dataset, the system:

1. Extracts task-specific features from the dataset
2. Uses a neural network (LearnwareCAHeterogeneous) to score each candidate model
3. Ranks models by predicted performance
4. Selects the best-performing model(s)

**Key Features:**
- ✅ Supports 30+ diverse vision datasets
- ✅ Ranks 72+ pre-trained models (homogeneous & heterogeneous architectures)
- ✅ Multi-head attention mechanism for model token selection
- ✅ Hierarchical model grouping (clusters c0-c95+)
- ✅ Evaluation metrics: weighted-tau & Pearson correlation

---

## Installation

### 1. Set up the Conda environment:

```bash
conda create --name spider-env python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate spider-env
```

### 2. Clone and install:

```bash
git clone https://github.com/zhangyikaii/Model-Spider.git
cd Model-Spider
pip install -r requirements.txt
```

### 3. Configure data path:

Choose your storage directory and configure it:

```bash
source ./scripts/modify-path.sh /path/to/your/storage
```

This sets the `PATH_TO_SRC_DATA` and other environment variables.

### 4. Download pre-trained model:

Download the pre-trained Model-Spider checkpoint from the [official repository](https://github.com/zhangyikaii/Model-Spider.git) and place it in your configured storage path.

- File: `best.pth`
- Size: ~100MB
- Location: `{storage_path}/best.pth`

---

## Quick Start

### Run Pre-trained Model (Inference Only)

```bash
bash scripts/test-model-spider.sh /path/to/storage/best.pth
```

This evaluates the pre-trained Model-Spider on test datasets and displays results.

**Expected Output:**
```
Model Spider's scores on ['resnet50', 'vit_base', 'swin_base', ...]
[0.9234, 0.7856, 0.8123, ...]
best heterogeneous_sample_num: 5
wtau of CIFAR10: 0.8956
wtau of Aircraft: 0.7834
...
```

---

## How to Run

### Basic Training Command

```bash
python trainer.py \
    --seed 0 \
    --train_dataset c86 c59 c16 c14 c38 c43 c9 c12 c32 c19 c31 c57 c29 \
    --val_dataset c86 \
    --test_dataset CIFAR10 Caltech101 DTD Pet Aircraft CIFAR100 Cars SUN397 dSprites \
    --test_size_threshold 1536 \
    --data_sub_url swin_base_7_checkpoint \
    --heterogeneous \
    --lr 0.00025 \
    --weight_decay 0.0005 \
    --momentum 0.5 \
    --max_epoch 30 \
    --optimizer Adam \
    --num_learnware 10 \
    --batch_size 16 \
    --dataset_size_threshold 5120 \
    --lr_scheduler cosine \
    --val_ratio 0.05 \
    --fixed_gt_size_threshold 64 \
    --heterogeneous_sampled_maxnum 10 \
    --data_url {storage_path}/data \
    --log_url {storage_path}/logs
```

### Inference on Pre-trained Model

```bash
python trainer.py \
    --seed 0 \
    --test_dataset CIFAR10 Caltech101 DTD Pet Aircraft \
    --data_url {storage_path}/data \
    --log_url {storage_path}/logs \
    --pretrained_url {storage_path}/best.pth
```

### Key Arguments Reference

| Category | Argument | Type | Default | Description |
|----------|----------|------|---------|-------------|
| **Dataset** | `--train_dataset` | str list | None | Model cluster IDs to train on |
| | `--test_dataset` | str list | None | Downstream tasks (CIFAR10, Aircraft, etc.) |
| | `--val_dataset` | str list | None | Optional validation dataset |
| **Training** | `--max_epoch` | int | 50 | Number of training epochs |
| | `--batch_size` | int | 128 | Training batch size |
| | `--lr` | float | 0.01 | Learning rate |
| | `--weight_decay` | float | 0.00005 | L2 regularization |
| | `--optimizer` | str | Adam | Adam or SGD |
| | `--lr_scheduler` | str | cosine | cosine, step, multistep, plateau |
| **Model** | `--num_learnware` | int | 72 | Number of model tokens to rank |
| | `--heterogeneous` | flag | False | Enable heterogeneous model support |
| | `--heterogeneous_sampled_maxnum` | int | 10 | Max heterogeneous models to sample |
| **Paths** | `--data_url` | str | '' | **Required**: Path to dataset features |
| | `--log_url` | str | logs | Directory to save logs & checkpoints |
| | `--pretrained_url` | str | None | **Optional**: Path to pre-trained model |
| **System** | `--gpu` | str | 0 | GPU device ID |
| | `--seed` | int | 1 | Random seed |
| | `--num_workers` | int | 8 | DataLoader workers |

For complete argument list, see [README_EXECUTION.md](./README_EXECUTION.md).

---

## Understanding the Program Flow

### High-Level Architecture

```
Input: Downstream task dataset
         ↓
  Feature Extraction (Swin backbone)
         ↓
  LearnwareCAHeterogeneous Model
  ├─ Multi-head Attention
  ├─ Heterogeneous Linear Layers
  └─ Model Token Embeddings
         ↓
  Output: Ranking scores for each PTM
         ↓
  Select & Evaluate Best Model(s)
```

### Training Loop (Per Epoch)

```
1. TRAINING PHASE:
   For each batch:
     - Forward pass (compute model suitability scores)
     - Loss computation (HierarchicalCE)
     - Backward pass & weight update

2. EVALUATION PHASE (k=0):
   Test on downstream tasks without heterogeneous models
   Measure: weighted-tau, Pearson correlation

3. EVALUATION PHASE (k=1..max):
   Test with k sampled heterogeneous models
   Find optimal k value

4. LOGGING:
   - Save model checkpoint
   - Save CSV results
   - TensorBoard metrics
   - Update learning rate
```

For detailed flow diagram and phase descriptions, see [README_EXECUTION.md](./README_EXECUTION.md).

---

## Output & Results

### Directory Structure

```
{log_url}/{setting_str}/{time_str}/
├─ 1.pth, 2.pth, ..., 30.pth     # Model checkpoints per epoch
├─ configs.json                   # Hyperparameter configuration
├─ scalars.json                   # Training metrics
├─ train.log                       # Training log file
├─ heterogeneous_sampled_acc.csv  # Per-epoch evaluation results
└─ tflogger/                       # TensorBoard event files
```

### Metrics Explanation

- **weighted-tau**: Weighted Kendall's tau correlation between predicted and actual model rankings
- **pearsonr**: Pearson correlation coefficient
- **k value**: Number of heterogeneous models sampled (larger k = more models considered)

### Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir {storage_path}/logs --port 6006
# Open: http://localhost:6006
```

---

## Baseline Methods

### Reproduce Baseline Results

The original Model-Spider paper evaluated several baseline methods. Results are stored in `assets/baseline_results.csv`.

To reproduce baseline method results:

```bash
# Ensure test datasets are in {storage_path}/data/
bash scripts/reproduce-baseline-methods.sh
```

**Baseline methods included:**
- H-Score
- LEEP
- LogME
- NCE
- OTCE
- PACTran
- GBC
- LFC

These scripts extract features and compute rankings using traditional model selection metrics (no learning).

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `--batch_size` or `--num_learnware` |
| Slow training | Increase `--num_workers`, use mixed precision |
| Loss not decreasing | Try different `--lr` values (0.0001 - 0.001) |
| File not found | Check `--data_url` and `--pretrained_url` paths |
| "Path to data not found" | Run `source ./scripts/modify-path.sh` with correct path |

---

## Datasets Supported

The system has been tested on 30+ vision datasets:

**Classification:** CIFAR-10, CIFAR-100, ImageNet, Caltech-101, Cars, CUB-2011, Dogs, Flowers, etc.

**Fine-grained:** Aircraft, DTD, Food-101, Oxford Pets, RESISC-45, AID, etc.

**Domain Adaptation:** PACS, OfficeHome, DomainNet, etc.

**Other:** EuroSAT, SmallNORB, STL-10, SUN397, SVHN, UTKFace, etc.

---

## Citation

If you use this code, please cite the original Model-Spider paper:

```bibtex
@inproceedings{zhang2023model,
  title={Model Spider: Learning to Rank Pre-trained Models Efficiently},
  author={Zhang, Yikang and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```

---

## References

- **Original Repository:** [Model-Spider](https://github.com/zhangyikaii/Model-Spider.git)
- **Paper:** Model Spider: Learning to Rank Pre-trained Models Efficiently
- **Execution Guide:** See [README_EXECUTION.md](./README_EXECUTION.md) for detailed program flow and arguments

---

## Future Enhancements

- [ ] Hierarchical clustering of model tokens for faster ranking
- [ ] Support for larger model pools (100+ models)
- [ ] Multi-task learning across different domains
- [ ] Distillation for lightweight ranking model
