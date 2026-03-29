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

### Quick Setup (Automated)

This project is now **fully self-contained** with all required files and models included in the repository.

**Step 1: Clone and enter directory**
```bash
git clone <repo-url>
cd Model-Nested-Spider
```

**Step 2: Create Conda environment with PyTorch**
```bash
conda create --name spider-env python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate spider-env
```

**Step 3: Install Python dependencies**
```bash
pip install -r requirements.txt
```

That's it! ✅ All data files (8GB of pre-trained models) and the best.pth checkpoint are already in the `data/` directory.

### Environment Configuration

Environment variables are automatically loaded from the included `.env` file:

```bash
# .env file (automatically loaded)
PATH_TO_SRC_DATA=./data
PATH_TO_LOG=./logs
PATH_TO_PRETRAINED_MODEL=./data
```

No manual path configuration needed! 🎉

---

## Quick Start

### Option 1: Use Run Script (Recommended)

**Linux/macOS:**
```bash
./run.sh
```

**Windows:**
```cmd
run.bat
```

This runs the default configuration (CIFAR10) with sensible defaults.

### Option 2: Custom Parameters

**Linux/macOS:**
```bash
./run.sh --train_dataset Dogs --test_dataset Flowers --max_epoch 20
```

**Windows:**
```cmd
run.bat --train_dataset Dogs --test_dataset Flowers --max_epoch 20
```

### Option 3: Direct Python Execution

```bash
python trainer.py \
    --train_dataset CIFAR10 \
    --test_dataset CIFAR10 \
    --max_epoch 50 \
    --batch_size 128
```

### Inference Mode (Using Pre-trained Model)

The repository includes `best.pth` (279MB) so you can immediately run inference:

**Linux/macOS:**
```bash
PRETRAINED_URL=./data/best.pth ./run.sh --test_dataset CIFAR10 Aircraft Flowers
```

**Windows:**
```cmd
set PRETRAINED_URL=./data/best.pth
run.bat --test_dataset CIFAR10 Aircraft Flowers
```

**Direct Python:**
```bash
python trainer.py \
    --test_dataset CIFAR10 Aircraft Flowers \
    --pretrained_url ./data/best.pth
```

---

## Complete Argument Reference

For comprehensive argument documentation, see [README_EXECUTION.md](./README_EXECUTION.md).

### Quick Reference

**Common Training Command:**
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
