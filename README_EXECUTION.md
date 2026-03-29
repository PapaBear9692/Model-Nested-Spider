# Model-Nested-Spider - Execution & Program Flow Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [How to Run](#how-to-run)
3. [Key Arguments](#key-arguments)
4. [Program Flow Overview](#program-flow-overview)
5. [Detailed Phase Breakdown](#detailed-phase-breakdown)
6. [Data Flow](#data-flow)
7. [Output Structure](#output-structure)
8. [Quick Reference](#quick-reference)

---

## Quick Start

### Easiest: Use Run Scripts

**Linux/macOS:**
```bash
./run.sh
```

**Windows:**
```cmd
run.bat
```

Both scripts automatically:
- ✅ Load environment from `.env` file
- ✅ Use sensible defaults (CIFAR10, batch_size=128, max_epoch=50)
- ✅ Display configuration before running
- ✅ Execute training with all paths configured

### Custom Parameters

**Linux/macOS:**
```bash
./run.sh --max_epoch 100 --batch_size 64 --train_dataset Dogs --test_dataset Flowers
```

**Windows:**
```cmd
run.bat --max_epoch 100 --batch_size 64 --train_dataset Dogs --test_dataset Flowers
```

### Set Environment Variables

**Linux/macOS:**
```bash
BATCH_SIZE=64 MAX_EPOCH=20 GPU=0 ./run.sh
```

**Windows:**
```cmd
set BATCH_SIZE=64
set MAX_EPOCH=20
set GPU=0
run.bat
```

---

## How to Run

### Pre-configured Environment

All paths are automatically configured via `.env` file:
```bash
# .env contents (automatically loaded)
PATH_TO_SRC_DATA=./data          # Model features directory
PATH_TO_LOG=./logs               # Logging & checkpoint output
PATH_TO_PRETRAINED_MODEL=./data  # Pre-trained model location
```

**No manual path configuration needed!** The environment loads automatically when Python starts.

### Option 1: Run Script (Recommended for Most Users)

**Default run (CIFAR10):**
```bash
./run.sh
```

**Custom parameters:**
```bash
./run.sh --max_epoch 100 --batch_size 64 --train_dataset Dogs --test_dataset Flowers
```

### Option 2: Direct Python Execution

**Basic training (auto-configured paths):**
```bash
python trainer.py \
    --train_dataset CIFAR10 \
    --test_dataset CIFAR10 \
    --max_epoch 50 \
    --batch_size 128
```

**Advanced training with heterogeneous models:**
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
    --heterogeneous_sampled_maxnum 10
```

### Option 3: Inference Mode (Using Pre-trained Model)

The repository includes `best.pth` (279MB) in the `data/` directory.

**Using run script:**
```bash
# Linux/macOS
PRETRAINED_URL=./data/best.pth ./run.sh --test_dataset CIFAR10 Aircraft Flowers

# Windows
set PRETRAINED_URL=./data/best.pth
run.bat --test_dataset CIFAR10 Aircraft Flowers
```

**Direct Python:**
```bash
python trainer.py \
    --test_dataset CIFAR10 Aircraft Flowers Pet SUN397 \
    --pretrained_url ./data/best.pth
```

When `--pretrained_url` is provided, the program **skips training** and runs inference only.

---

## Key Arguments

### Dataset Configuration
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--train_dataset` | str list | None | Model cluster IDs to train on (e.g., c86, c59, c16) |
| `--val_dataset` | str list | None | Optional validation dataset |
| `--test_dataset` | str list | None | Downstream tasks (CIFAR10, Aircraft, ImageNet, etc.) |
| `--test_size_threshold` | int | 1024 | Max samples per test dataset |
| `--dataset_size_threshold` | int | 0 | Limit training dataset size |
| `--val_ratio` | float | 0.2 | Validation split ratio |

### Training Configuration
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max_epoch` | int | 50 | Number of training epochs |
| `--batch_size` | int | 128 | Training batch size |
| `--lr` | float | 0.01 | Learning rate |
| `--weight_decay` | float | 0.00005 | L2 regularization coefficient |
| `--momentum` | float | 0.8 | SGD momentum |
| `--optimizer` | str | Adam | Adam or SGD |
| `--lr_scheduler` | str | cosine | Learning rate schedule: cosine, step, multistep, plateau |
| `--cosine_annealing_lr_eta_min` | float | 5e-6 | Minimum LR for cosine scheduler |

### Model Configuration
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_learnware` | int | 72 | Number of model tokens to rank |
| `--heterogeneous` | flag | False | Enable heterogeneous model support |
| `--heterogeneous_sampled_maxnum` | int | 10 | Max heterogeneous models to sample |
| `--heterogeneous_sampled_minnum` | int | 0 | Min heterogeneous models to sample |
| `--data_sub_url` | str | swin_base_7_checkpoint | Feature extractor backbone |
| `--attn_pool` | str | cls | Spatial pooling: cls or mean |
| `--fixed_gt_size_threshold` | int | 128 | Prototype dimension |

### Path Configuration
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_url` | str | ./data | Path to dataset features (auto-configured from .env) |
| `--log_url` | str | ./logs | Directory to save logs & checkpoints (auto-configured from .env) |
| `--pretrained_url` | str | None | **Optional**: Path to pre-trained model (enables inference mode) |

### System Configuration
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--gpu` | str | 0 | GPU device ID (e.g., "0,1" for multi-GPU) |
| `--seed` | int | 1 | Random seed for reproducibility |
| `--num_workers` | int | 8 | DataLoader worker processes |

---

## Program Flow Overview

```
main()
  ├─ Parse command-line arguments
  ├─ Setup GPU & fix random seed
  └─ Trainer(args)
      ├─ Build model: LearnwareCAHeterogeneous
      ├─ Load datasets (train/val/test)
      ├─ Create DataLoaders
      ├─ Setup optimizer & LR scheduler
      └─ trainer.fit()
          │
          ├─ [IF pretrained_url exists]
          │  └─ Inference-only mode: test() → exit
          │
          └─ [ELSE: Training mode]
             └─ For each epoch:
                ├─ TRAINING: Forward pass, compute loss, backward
                ├─ TEST (k=0): Predict rankings on test datasets
                ├─ TEST (k=1..max): Evaluate with k heterogeneous models
                ├─ LOG: Save checkpoint, CSV results, metrics
                └─ UPDATE: Adjust learning rate
```

---

## Detailed Phase Breakdown

### Phase 1: Initialization

```python
main() {
    1. Parse command-line arguments using argparse
    2. set_gpu(args.gpu)          # Enable CUDA on specified GPU
    3. set_seed(args.seed)        # Fix numpy, torch, python randomness
    4. Trainer(args)              # Initialize training system
}
```

### Phase 2: Trainer Setup (Constructor)

```python
Trainer.__init__(args) {
    1. Build neural network: LearnwareCAHeterogeneous
       - Multi-head attention mechanism for model tokens
       - Heterogeneous linear layers for different model families
       - Supports variable input lengths with padding masks
    
    2. Load 3 datasets:
       - Train dataset: Model clusters (c86, c59, etc.)
                       Used to train model ranking ability
       - Val dataset: Optional validation clusters
       - Test dataset: Real downstream tasks (CIFAR10, Aircraft, etc.)
                      Used to evaluate final performance
    
    3. Create PyTorch DataLoaders:
       - train_dataloader: Batched training data
       - val_dataloader: Batched validation data
       - test_dataloader: Batched test data
    
    4. Setup optimizer:
       - Adam: For adaptive learning rates
       - SGD: With momentum for stability
    
    5. Setup learning rate scheduler:
       - cosine: Gradually decrease LR (recommended)
       - step: Drop LR at specific epochs
       - multistep: Drop LR at multiple epochs
       - plateau: Reduce on validation plateau
    
    6. Setup loss function: HierarchicalCE
       - Cross-entropy loss for ranking
       - Encourages correct model ordering
    
    7. Check if pre-trained:
       - If --pretrained_url provided: do_train = False (inference only)
       - Otherwise: do_train = True (full training mode)
}
```

### Phase 3A: Training Mode (when do_train = True)

```
trainer.fit() {
    For epoch = 1 to max_epoch:
    
    ┌────────────────────────────────────────────────────────┐
    │ STEP 1: MODEL TRAINING                                 │
    ├────────────────────────────────────────────────────────┤
    
    Set model to training mode: model.train()
    
    For each batch in train_dataloader:
        1. Load batch data:
           - x_uni: Unified backbone features
                   Shape: [batch_size × feature_dim]
           - x_hete: Heterogeneous features from different families
                    Dict[model_family → List[features]]
           - labels: Ground truth model rankings
                    Shape: [batch_size × num_learnware]
                    Each value = ranking position of that model
           - pad_length: Actual sequence length
        
        2. Preprocess inputs:
           - Apply linear transformation to x_uni
           - Apply family-specific linear layers to x_hete
           - Create attention padding masks for variable lengths
        
        3. Forward pass:
           outputs = model(x_uni, x_hete, attention_masks)
           # outputs: [batch_size × num_learnware]
           # Each output[i][j] = suitability score of model j for task i
        
        4. Compute loss:
           loss = criterion(outputs, labels)
           # HierarchicalCE: rank loss encouraging correct ordering
        
        5. Backward pass & optimization:
           optimizer.zero_grad()    # Clear old gradients
           loss.backward()          # Compute gradients
           optimizer.step()         # Update weights
        
        6. Log training metrics:
           Log loss to TensorBoard and JSON files
    
    ┌────────────────────────────────────────────────────────┐
    │ STEP 2: EVALUATION (k=0, no heterogeneous models)      │
    ├────────────────────────────────────────────────────────┤
    
    Set model to evaluation mode: model.eval()
    
    For each batch in test_dataloader:
        1. Forward pass (no gradients):
           outputs = model(x_uni, x_hete, masks)
        
        2. Convert outputs to rankings:
           rankings = torch.argsort(outputs)
               # Smaller ranking = better model
        
        3. Compute correlation metrics:
           - weighted-tau: Kendall's tau correlation (weighted)
           - pearsonr: Pearson correlation coefficient
        
        4. Group results by test dataset:
           mt_raw_results[dataset] = list of output scores
           mt_results[dataset][metric] = average metric value
    
    Result: Know which models perform best for each test task
    
    ┌────────────────────────────────────────────────────────┐
    │ STEP 3: EVALUATION (k=1 to maxnum, heterogeneous)      │
    ├────────────────────────────────────────────────────────┤
    
    For sample_num = 1 to heterogeneous_sampled_maxnum:
        
        1. Prefetch: Extract top-performing models from k=0
           LearnwareDataset.__heterogeneous_prefetch_rank__ = mt_raw_results
        
        2. Configure: Set number of models to sample
           LearnwareDataset.__heterogeneous_sampled_fixnum__ = sample_num
        
        3. Run test:
           For each batch in test_dataloader:
               - Load batch with exactly sample_num models
               - Forward pass
               - Compute metrics
        
        4. Store results:
           mt_hete_results[dataset][sample_num] = metric_value
    
    Result: Performance curve showing effect of k
    
    ┌────────────────────────────────────────────────────────┐
    │ STEP 4: LOGGING & CHECKPOINTING                        │
    ├────────────────────────────────────────────────────────┤
    
    1. Save model checkpoint:
       torch.save({model, optimizer, epoch}, f'{epoch}.pth')
    
    2. Save CSV results:
       heterogeneous_sampled_acc.csv
       Format: epoch,dataset_name,k0_metric,k1_metric,k2_metric,...
       Example:
         1,CIFAR10,0.845,0.852,0.878,0.891
         1,Aircraft,0.723,0.741,0.756,0.768
    
    3. Track best performance:
       If current epoch surpasses previous best:
           best_state[dataset][metric] = current_result
           Log: "NEW BEST: epoch X, dataset Y, metric Z"
    
    4. TensorBoard logging:
       Add scalar values for training loss,
       validation metrics per epoch
    
    ┌────────────────────────────────────────────────────────┐
    │ STEP 5: LEARNING RATE UPDATE                           │
    ├────────────────────────────────────────────────────────┤
    
    lr_scheduler.step()    # Proceed to next epoch's LR
    
}  # End: For each epoch
```

### Phase 3B: Inference Mode (when --pretrained_url provided)

```python
trainer.fit() {
    # Skip all training steps
    
    epoch = 0
    
    1. Load pre-trained model:
       state_dict = torch.load(args.pretrained_url)
       model.load_state_dict(state_dict['model'])
       model.eval()
    
    2. Test Phase k=0 (no heterogeneous models):
       For each test batch:
           outputs = model(x_uni, x_hete)
           Store rankings
       
       For each test_dataset:
           Sort models by predicted quality
           Store top-K ranking in mt_raw_results
    
    3. Test Phase k=1 to maxnum:
       LearnwareDataset.__heterogeneous_prefetch_rank__ = mt_raw_results
       
       For k = 1 to heterogeneous_sampled_maxnum:
           Set sample_num = k
           Run test forward passes
           Compute metrics
           Store in mt_hete_results[dataset][k]
    
    4. Find optimal k:
       best_k = argmax(average_metric_across_datasets)
       LearnwareDataset.__heterogeneous_sampled_fixnum__ = best_k
    
    5. Output results:
       Print model suitability scores:
           Model Spider's scores on [model_A, model_B, ...]
           [0.89, 0.75, 0.82, ...]
       
       Print best k value:
           best heterogeneous_sample_num: 5
       
       Print metrics per dataset:
           wtau of CIFAR10: 0.876
           wtau of Aircraft: 0.734
           ...
    
    6. Exit:
       return  # Program terminates
}
```

---

## Data Flow

### Input Data Structure

```
Training/Test Batch:
├─ x_uni [batch_size × feature_dim]
│  └─ Unified features from backbone (Swin, ResNet, etc.)
│     Shape example: [16 × 768]
│
├─ x_hete [Dict[model_family → List[features]]]
│  └─ Features from different model architectures
│  └─ Example:
│     {
│       'resnet': [feat_batch_1, feat_batch_2, ...],
│       'vit': [feat_batch_1, feat_batch_2, ...],
│       'efficient': [feat_batch_1, feat_batch_2, ...]
│     }
│
├─ labels [batch_size × num_learnware]
│  └─ Ground truth rankings
│  └─ Example: [[2, 5, 1, 8, ...],  # Sample 1: model 0 is rank 2
│               [1, 3, 2, 7, ...]]  # Sample 2: model 0 is rank 1
│
└─ pad_length [batch_size]
   └─ Actual token count (for masking padding)
   └─ Example: [100, 128, 95, ...]
```

### Model Processing Pipeline

```
LearnwareCAHeterogeneous Forward Pass:
  
  Input: x_uni, x_hete, pad_length, attn_masks
    │
    ├─ Step 1: Linear transformation
    │  ├─ x_uni_transformed = self.uni_linear(x_uni)
    │  │  └─ Shape: [batch_size × dim]
    │  │
    │  └─ x_hete_dict = {bkb: self.hete_linears[bkb](x_hete[bkb]) 
    │                      for bkb in x_hete}
    │     └─ Align all families to same dimension
    │
    ├─ Step 2: Concatenate features
    │  └─ x_combined = [x_uni_transformed; x_hete_dict[0]; x_hete_dict[1]; ...]
    │     └─ Shape: [batch_size × (dim + heterogeneous_dim)]
    │
    ├─ Step 3: Multi-head attention
    │  ├─ Q (Query): Task/dataset representation
    │  ├─ K (Key): Model token embeddings
    │  ├─ V (Value): Model token embeddings
    │  ├─ Attention Mask: [0 for valid, 1 for padding]
    │  │  └─ Prevents attention to padding positions
    │  │
    │  └─ attn_output = MultiHeadAttention(Q, K, V, mask)
    │     └─ Shape: [batch_size × num_learnware × dim]
    │
    └─ Step 4: Output projection
       └─ outputs = self.output_layer(attn_output)
          └─ Shape: [batch_size × num_learnware]
          └─ Each value = suitability score for that model

Output: scores for each model [0.89, 0.45, 0.92, ...]
        ↓
Convert to rankings (lower score = lower rank):
        rankings = [2, 0, 3, 1, ...]
        ↓
Compare with ground truth rankings:
        similarity = correlation(predicted_rankings, true_rankings)
```

### Loss Computation

```
HierarchicalCE (Hierarchical Cross-Entropy):
  
  Input: outputs [batch_size × num_learnware]
         labels  [batch_size × num_learnware]
  
  Process:
    1. Interpret outputs as predicted suitability scores
    2. Interpret labels as ground-truth suitability
    3. Compute ranking loss that encourages:
       - Better models get higher scores
       - Worse models get lower scores
       - Relative model ordering is correct
  
  Output: scalar loss value
           ↓
  Backward: Compute gradients w.r.t. all model parameters
```

---

## Output Structure

### Directory Layout

```
$log_url/{setting_str}/{time_str}/
├─ {epoch}.pth                      # Model checkpoint
│  └─ Contains: model state dict, optimizer state, epoch number
│
├─ configs.json                     # Hyperparameter configuration
│  └─ JSON dump of all command-line arguments
│
├─ scalars.json                     # Training metrics over time
│  └─ {"epoch_loss": {"1": 0.456, "2": 0.345, ...},
│      "other_metrics": {...}}
│
├─ train.log                        # Console output log file
│  └─ All logging messages written to console
│
├─ heterogeneous_sampled_acc.csv    # Per-epoch evaluation results
│  └─ Format:
│     epoch,dataset_name,k0,k1,k2,k3,...
│     1,CIFAR10,0.845,0.852,0.878,0.891
│     1,Aircraft,0.723,0.741,0.756,0.768
│     2,CIFAR10,0.851,0.865,0.889,0.902
│     2,Aircraft,0.728,0.752,0.771,0.789
│
└─ tflogger/                        # TensorBoard event files
   └─ events.out.tfevents...        # Binary TensorBoard logs
      └─ View with: tensorboard --logdir [this_folder]
```

### Example Output

```
[TRAINING LOG]
03-29 14:32:45 [1 / 30] lr: 0.00025
03-29 14:45:12 [1] CIFAR10 test      weightedtau  0.8451  pearsonr  0.8234
03-29 14:45:12 [1] Aircraft test     weightedtau  0.7234  pearsonr  0.7012
...
03-29 17:22:18 [5 / 30] lr: 0.00015
03-29 17:35:45 [5] CIFAR10 test      weightedtau  0.8934  pearsonr  0.8812
03-29 17:35:45 [5] Aircraft test     weightedtau  0.7856  pearsonr  0.7645
...

[INFERENCE LOG]
Model Spider's scores on ['resnet50', 'vit_base', 'swin_base', ...]
[0.9234, 0.8123, 0.7456, ...]
best heterogeneous_sample_num: 5
wtau of CIFAR10: 0.8956
wtau of Aircraft: 0.7834
...
```

---

## Quick Reference

### Environment Setup

**View environment configuration:**
```bash
cat .env
```

**Output:**
```
PATH_TO_SRC_DATA=./data
PATH_TO_LOG=./logs
PATH_TO_PRETRAINED_MODEL=./data
```

All paths are **automatically loaded** - no manual setup required!

### Common Commands

**1. Train with default settings (recommended):**
```bash
# Linux/macOS
./run.sh

# Windows
run.bat
```

**2. Train with custom datasets:**
```bash
# Linux/macOS
./run.sh --train_dataset Dogs Flowers --test_dataset CIFAR10 Aircraft

# Windows
run.bat --train_dataset Dogs Flowers --test_dataset CIFAR10 Aircraft
```

**3. Direct Python with custom settings:**
```bash
python trainer.py \
    --train_dataset c86 c59 c16 \
    --test_dataset CIFAR10 Aircraft \
    --max_epoch 50 \
    --batch_size 128
```

**4. Run inference on pre-trained model:**
```bash
# Linux/macOS
PRETRAINED_URL=./data/best.pth ./run.sh --test_dataset CIFAR10 Aircraft

# Windows
set PRETRAINED_URL=./data/best.pth
run.bat --test_dataset CIFAR10 Aircraft

# Or direct Python
python trainer.py \
    --test_dataset CIFAR10 Aircraft DTD \
    --pretrained_url ./data/best.pth
```

**5. Monitor training with TensorBoard:**
```bash
tensorboard --logdir ./logs --port 6006
# Open browser: http://localhost:6006
```

**6. Check output results:**
```bash
# View latest training results
ls -lt ./logs/[setting_str]/[latest_time_str]/

# Check evaluation metrics
cat ./logs/[setting_str]/[latest_time_str]/heterogeneous_sampled_acc.csv
```

### Troubleshooting

**Issue: "dotenv not found"**
```bash
pip install python-dotenv
```

**Issue: "CUDA out of memory"**
```bash
./run.sh --batch_size 32   # Reduce batch size
```

Or set via environment:
```bash
BATCH_SIZE=32 ./run.sh
```

**Issue: ".env file not found"**
```bash
# The .env file should be in the repo root
# Check it exists:
cat .env

# Or recreate from template:
cat > .env << EOF
PATH_TO_SRC_DATA=./data
PATH_TO_LOG=./logs
PATH_TO_PRETRAINED_MODEL=./data
EOF
```

**Issue: Missing data files**
```bash
# Verify data directory structure
ls -la ./data/implclproto/ | head -10

# Check best.pth exists
ls -lh ./data/best.pth
```

---

## Advanced Usage

### Resume training from checkpoint:
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `--batch_size` (e.g., 128 → 64) |
| Slow training | Increase `--num_workers` (e.g., 8 → 16) |
| Loss not decreasing | Increase `--lr` (e.g., 0.00025 → 0.001) |
| Diverging loss (NaN) | Decrease `--lr`, increase `--weight_decay` |
| File not found error | Check `--data_url` and `--log_url` paths |

### Key Hyperparameters to Tune

| Parameter | Effect | Typical Range |
|-----------|--------|----------------|
| `--lr` | Learning rate | 0.0001 - 0.01 |
| `--weight_decay` | L2 regularization | 0.0 - 0.001 |
| `--batch_size` | Batch size | 8 - 256 |
| `--max_epoch` | Training duration | 10 - 100 |
| `--heterogeneous_sampled_maxnum` | Max models to sample | 1 - 20 |

---

## Next Steps: Your Enhancement Plan

**Goal:** Add hierarchical clustering of model tokens to speed up best model selection.

**Current System:**
- Model pool is flat: 72 models treated equally
- Ranking happens in single stage

**Proposed Enhancement:**
- Cluster similar models hierarchically
- Two-stage ranking:
  1. Coarse: Which cluster has best models?
  2. Fine: Best model within selected cluster?
- **Benefit:** Reduce inference search space by 2-4x

**Implementation Points:**
- Modify `LearnwareCAHeterogeneous` model
- Add clustering layer to model tokens
- Modify ranking loss to account for hierarchical structure
- Update evaluation metrics to measure speedup
