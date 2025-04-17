# Configuration System for ImprovedPerceiver

This directory contains YAML configuration files for training Perceiver models. The configuration system is designed to be modular, with logical grouping of parameters into sections.

## Configuration Structure

Each YAML configuration file is organized into the following sections:

### Model Architecture
```yaml
model:
  input_dim: 37               # Input dimension
  depth: 1                    # Number of cross-attention layers
  output_dim: 100             # Output dimension (number of classes)
  num_latents: 256            # Number of latent vectors
  latent_dim: 256             # Dimension of latent vectors
  cross_heads: 1              # Number of cross-attention heads
  cross_head_dim: 64          # Dimension of cross-attention heads
  cross_rotary_emb_dim: 0     # Dimension of cross-attention rotary embeddings
  cross_attn_dropout: 0.1     # Dropout rate for cross-attention
  latent_heads: 8             # Number of latent self-attention heads
  latent_head_dim: 64         # Dimension of latent self-attention heads
  latent_rotary_emb_dim: 0    # Dimension of latent self-attention rotary embeddings
  latent_attn_dropout: 0.1    # Dropout rate for latent self-attention
  latent_drop: 0.1            # Dropout rate for latent vectors
  weight_tie_layers: true     # Whether to share weights across layers
  gated_mlp: false            # Whether to use gated MLPs
  self_per_cross_attn: 6      # Number of self-attention blocks per cross-attention
  output_mode: "average"      # Output mode: "average", "concat", or "first"
  num_zero_tokens: null       # Number of zero tokens (null for none)
  use_flash_attn: true        # Whether to use FlashAttention
```

### Positional Encoding
```yaml
pos_encoding:
  bands: 8                    # Number of Fourier bands
  dims: 2                     # Dimensions of positions (2 for 2D images)
  max_res: 32                 # Maximum resolution (typically image width/height)
```

### Training Parameters
```yaml
training:
  lr: 0.0005                  # Learning rate
  weight_decay: 0.001         # Weight decay for optimizer
  batch_size: 1024            # Batch size
  epochs: 400                 # Number of epochs
  device: "cuda"              # Device to use (cuda or cpu)
  data_root: "../data"        # Data directory
  checkpoint_interval: 50     # Save checkpoint every N epochs (0 to disable)
```

### Optimizer Configuration
```yaml
optimizer:
  type: "adamw"               # Optimizer type: "lamb" (default) or "adamw"
  betas: [0.9, 0.999]         # Beta coefficients for AdamW (only used for adamw)
  eps: 1.0e-8                 # Epsilon value for numerical stability (only used for adamw)
```

### Data Augmentation
```yaml
augmentation:
  use_randaug: true           # Whether to use RandAugment
  randaug_n: 1                # Number of augmentations to apply
  randaug_m: 5                # Magnitude of augmentations (0-30)
```

### Learning Rate Scheduler
```yaml
scheduler:
  type: "step"                # Scheduler type: "step" or "cosine"
  warmup_steps: 1000          # Number of warmup steps
  
  # Step scheduler parameters
  step_milestones: [4000, 11000]  # Steps at which to reduce LR
  step_gamma: 0.5                 # Multiplicative factor for LR reduction
  
  # Cosine scheduler parameters
  plateau_steps: 3000          # Number of steps to keep at peak LR after warmup
  min_lr_ratio: 0.1            # Minimum LR as fraction of base LR
  restart_steps: 500           # Steps for periodic restarts (0 = no restarts)
```

### Logging
```yaml
logging:
  use_tensorboard: true       # Whether to use TensorBoard
  logdir: "../logs"           # Directory for logs
```

## Using Configurations

To train a model with a specific configuration, use:

```bash
python examples/cifar_classification.py --config configs/cifar100.yaml
```

## Checkpoint System

The training script saves several types of checkpoints:

1. **Best Model Checkpoint**: The model with the highest validation accuracy
   - Named: `best_model_[runname]_epoch[N]_acc[X.XXXX].pt`
   - Also saved as: `best_model.pt` for easy loading

2. **Last Model Checkpoint**: The model after the final epoch
   - Named: `last_model_[runname]_epoch[N]_acc[X.XXXX].pt`
   - Also saved as: `last_model.pt` for easy loading

3. **Periodic Checkpoints**: Models saved at regular intervals (controlled by `checkpoint_interval`)
   - Named: `checkpoint_[runname]_epoch[N]_acc[X.XXXX].pt`

All checkpoints are saved in the run's specific log directory (e.g., `logs/cifar100_20250417-115749/`). 