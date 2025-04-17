# PyTorch Lightning Implementation for CIFAR-100 Classification

This is a PyTorch Lightning implementation of the CIFAR-100 classification training script using the Perceiver architecture. The implementation provides the same functionality as the original script but leverages Lightning's features for better organization and less boilerplate code.

## Files Overview

- `cifar_classification_lightning.py`: Main training script
- `cifar_perceiver_module.py`: Lightning module for the Perceiver model
- `cifar_data_module.py`: Lightning data module for CIFAR-100 dataset
- `custom_schedulers.py`: Custom learning rate schedulers (cosine with warmup and step with warmup)
- `configs/cifar_lightning.yaml`: Sample configuration file

## Features

- **Same functionality** as the original script
- Support for **gradient accumulation**
- **Learning rate schedulers** (cosine with warmup, step with warmup)
- **Data augmentation** options (standard, RandAugment, MoCo-style)
- **TensorBoard logging** and **checkpointing**
- **Automatic mixed precision** training on GPUs
- **Model and FLOPs statistics** calculation

## Usage

To train a model:

```bash
python cifar_classification_lightning.py --config configs/cifar_lightning.yaml
```

## Configuration

The configuration file follows the same structure as the original implementation. Example:

```yaml
# Device configuration
device: "cuda"  # Use "cuda" for GPU, "cpu" for CPU

# Model configuration
model:
  input_dim: 37
  depth: 1
  # ...other model parameters...

# Training configuration
training:
  lr: 5.0e-4
  batch_size: 256
  # ...other training parameters...

# Augmentation configuration
augmentation:
  type: "randaug"  # "standard", "randaug", or "moco"
  # ...augmentation parameters...
```

See `configs/cifar_lightning.yaml` for a complete configuration example.

## Key Advantages Over Original Implementation

1. **Code Organization**: Cleaner separation of model, data, and training logic
2. **Less Boilerplate**: Training loops and validation handled by Lightning
3. **Features Built-in**: Gradient accumulation, mixed precision, checkpointing
4. **Easier Debugging**: Well-structured components are easier to debug
5. **Scalability**: Easy to scale to multi-GPU setups with minimal code changes 