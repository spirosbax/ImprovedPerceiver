import sys
sys.path.append("..")
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR100
from pytorch_lamb import Lamb
from tqdm import tqdm
from flash_perceiver import utils
from flash_perceiver.perceiver import Perceiver
from flash_perceiver.adapters import ImageAdapter
from flash_perceiver.utils.encodings import PerceiverPositionalEncoding
from flash_perceiver.utils.training import ImprovedCosineWithWarmupLR, WarmupStepLR
from flash_perceiver.utils.config import ConfigManager
from torch.utils.tensorboard import SummaryWriter
import os
import yaml
from datetime import datetime
import torchmetrics
import numpy as np

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="CIFAR100 Training with Perceiver")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to YAML config file")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    
    # Extract configurations
    model_config = config_manager.get_model_config()
    pos_encoding_config = config_manager.get_pos_encoding_config()
    training_config = config_manager.get_training_config()
    augmentation_config = config_manager.get_augmentation_config()
    scheduler_config = config_manager.get_scheduler_config()
    logging_config = config_manager.get_logging_config()
    
    # Set device
    device = config_manager.get_device()
    
    # Initialize TensorBoard writer if enabled
    writer = None
    if logging_config.get('use_tensorboard', True):
        log_path = os.path.join(
            logging_config.get('logdir', './logs'), 
            f'cifar100_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        )
        writer = SummaryWriter(log_path)
        print(f"TensorBoard logs will be saved to {log_path}")
        
        # Log the configuration
        writer.add_text("config/yaml", str(config_manager.config), 0)
        
        # Save the config to the log directory
        config_manager.save_config(os.path.join(log_path, "config.yaml"))
    
    print("==> Preparing data..")
    
    # Define transforms with optional RandAugment
    transform_train = []
    if augmentation_config.get('use_randaug', False):
        randaug_n = augmentation_config.get('randaug_n', 2)
        randaug_m = augmentation_config.get('randaug_m', 9)
        print(f"==> Using RandAugment with N={randaug_n}, M={randaug_m}")
        transform_train.append(transforms.RandAugment(num_ops=randaug_n, magnitude=randaug_m))
    
    # Add standard augmentations
    transform_train.extend([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_train = transforms.Compose(transform_train)
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load datasets
    data_root = training_config.get('data_root', '../data')
    batch_size = training_config.get('batch_size', 512)
    
    train_set = CIFAR100(data_root, train=True, download=True, transform=transform_train)
    test_set = CIFAR100(data_root, train=False, download=True, transform=transform_test)
    
    valid_set, train_set = random_split(train_set, [0.1, 0.9])
    
    train_batches = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_batches = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_batches = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print("==> Building model..")
    
    # Initialize model
    model = nn.Sequential(
        ImageAdapter(
            embed_dim=None,
            pos_encoding=PerceiverPositionalEncoding(
                pos_encoding_config.get('dims', 2), 
                n_bands=pos_encoding_config.get('bands', 8), 
                max_res=pos_encoding_config.get('max_res', 32), 
                concat_pos=True
            ),
        ),
        Perceiver(
            input_dim=model_config.get('input_dim', 37),
            depth=model_config.get('depth', 1),
            output_dim=model_config.get('output_dim', 100),
            num_latents=model_config.get('num_latents', 512),
            latent_dim=model_config.get('latent_dim', 256),
            cross_heads=model_config.get('cross_heads', 1),
            cross_head_dim=model_config.get('cross_head_dim', 64),
            cross_rotary_emb_dim=model_config.get('cross_rotary_emb_dim', 0),
            cross_attn_dropout=model_config.get('cross_attn_dropout', 0.0),
            latent_heads=model_config.get('latent_heads', 8),
            latent_head_dim=model_config.get('latent_head_dim', 64),
            latent_rotary_emb_dim=model_config.get('latent_rotary_emb_dim', 0),
            latent_attn_dropout=model_config.get('latent_attn_dropout', 0.0),
            latent_drop=model_config.get('latent_drop', 0.0),
            weight_tie_layers=model_config.get('weight_tie_layers', False),
            gated_mlp=model_config.get('gated_mlp', True),
            self_per_cross_attn=model_config.get('self_per_cross_attn', 1),
            output_mode=model_config.get('output_mode', 'average'),
            num_zero_tokens=model_config.get('num_zero_tokens', None),
            use_flash_attn=model_config.get('use_flash_attn', True),
        ),
    ).to(device)
    
    # Helper function to calculate model size and estimated FLOPs
    def count_parameters_and_flops(model, input_shape=(1, 3, 32, 32)):
        """Count parameters and estimate FLOPs for the model"""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Get model attributes for FLOP calculation
        if hasattr(model, 'num_latents') and hasattr(model, 'latent_dim'):
            # Direct access to model attributes
            L = model.num_latents
            D = model.latent_dim
        else:
            # Access through submodule (for Sequential model)
            for module in model.modules():
                if hasattr(module, 'num_latents') and hasattr(module, 'latent_dim'):
                    L = module.num_latents
                    D = module.latent_dim
                    break
            else:
                # Fallback to config values
                L = model_config.get('num_latents', 512)
                D = model_config.get('latent_dim', 256)
        
        # Image adapter converts 32x32x3 image to sequence of length N
        H, W = input_shape[2], input_shape[3]
        patch_size = 1  # Default assumption for Perceiver
        N = (H // patch_size) * (W // patch_size)
        
        # Estimate FLOPs for different components
        depth = model_config.get('depth', 1)
        self_per_cross_attn = model_config.get('self_per_cross_attn', 1)
        
        # Cross-attention FLOPs (once per layer)
        cross_attn_flops = depth * 2 * L * N * D
        
        # Self-attention FLOPs (self_per_cross_attn times per layer)
        self_attn_blocks = depth * self_per_cross_attn
        self_attn_flops = self_attn_blocks * 4 * L * L * D
        
        # MLP FLOPs (twice per attention block - both for cross and self attention)
        mlp_blocks = depth * (1 + self_per_cross_attn)
        mlp_flops = mlp_blocks * 8 * L * D * D  # Assuming 4x expansion in MLP
        
        # Total estimated FLOPs for a forward pass
        total_flops = cross_attn_flops + self_attn_flops + mlp_flops
        
        return {
            "params": total_params,
            "flops": total_flops,
            "flops_G": total_flops / 1e9,
            "flops_per_param": total_flops / total_params
        }
    
    # Calculate and print FLOPS and parameters
    compute_metrics = count_parameters_and_flops(model)
    
    # Store compute metrics for logging
    compute_metrics_dict = {
        "model_params": compute_metrics["params"],
        "model_flops": compute_metrics["flops"],
        "model_flops_G": compute_metrics["flops_G"],
        "flops_per_param": compute_metrics["flops_per_param"]
    }
    
    print(f"Number of parameters: {compute_metrics['params']:,}")
    print(f"Estimated FLOPs: {compute_metrics['flops']:,} ({compute_metrics['flops_G']:.2f} GFLOPS)")
    print(f"FLOPs per parameter: {compute_metrics['flops_per_param']:.2f}")
    
    # Initialize criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Lamb(
        model.parameters(), 
        lr=training_config.get('lr', 5e-4), 
        weight_decay=training_config.get('weight_decay', 1e-4)
    )
    
    # Configure learning rate scheduler
    scheduler_type = scheduler_config.get('type', 'cosine')
    warmup_steps = scheduler_config.get('warmup_steps', 1000)
    
    if scheduler_type == "step":
        step_milestones = scheduler_config.get('step_milestones', [15000, 30000])
        step_gamma = scheduler_config.get('step_gamma', 0.5)
        
        print("\n==> Using step-based learning rate scheduler with:")
        print(f"    - {warmup_steps} warmup steps")
        print(f"    - LR reduction by factor of {step_gamma} at steps: {step_milestones}")
        print("")
        
        scheduler = WarmupStepLR(
            optimizer,
            warmup_steps=warmup_steps,
            milestones=step_milestones,
            gamma=step_gamma
        )
    else:  # Default to cosine scheduler
        plateau_steps = scheduler_config.get('plateau_steps', 2000)
        min_lr_ratio = scheduler_config.get('min_lr_ratio', 0.1)
        restart_steps = scheduler_config.get('restart_steps', 0)
        
        print("\n==> Using enhanced cosine learning rate scheduler with:")
        print(f"    - {warmup_steps} warmup steps")
        print(f"    - {plateau_steps} plateau steps after warmup")
        print(f"    - Min LR ratio of {min_lr_ratio} (LR never drops below {training_config.get('lr', 5e-4) * min_lr_ratio:.2e})")
        if restart_steps > 0:
            print(f"    - SGDR warm restarts every {restart_steps} steps")
        else:
            print("    - No warm restarts (standard cosine decay)")
        print("")
        
        scheduler = ImprovedCosineWithWarmupLR(
            optimizer, 
            training_steps=training_config.get('epochs', 100) * len(train_batches), 
            warmup_steps=warmup_steps,
            plateau_steps=plateau_steps,
            min_lr_ratio=min_lr_ratio,
            restart_steps=restart_steps
        )
    
    # Print data augmentation information
    if augmentation_config.get('use_randaug', False):
        print("==> Using RandAugment for training data:")
        print(f"    - N = {augmentation_config.get('randaug_n', 2)} (number of augmentations applied sequentially)")
        print(f"    - M = {augmentation_config.get('randaug_m', 9)} (magnitude of augmentations, range 0-30)")
        print("")
    else:
        print("==> Using standard data augmentation (no RandAugment)")
        print("")
    
    # Initialize metrics
    accuracy_top1 = torchmetrics.Accuracy(task="multiclass", num_classes=100, top_k=1).to(device)
    accuracy_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=100, top_k=5).to(device)
    accuracy_top10 = torchmetrics.Accuracy(task="multiclass", num_classes=100, top_k=10).to(device)
    
    # Helper function to save model checkpoints
    def save_checkpoint(checkpoint_path, epoch, acc1, acc5=None, acc10=None, is_best=False, is_last=False, is_periodic=False):
        """
        Save a model checkpoint with detailed metrics in the filename
        
        Args:
            checkpoint_path: Path to save the checkpoint
            epoch: Current epoch
            acc1: Top-1 accuracy
            acc5: Top-5 accuracy (optional)
            acc10: Top-10 accuracy (optional)
            is_best: Whether this is the best model so far
            is_last: Whether this is the last model
            is_periodic: Whether this is a periodic save
        """
        # Format accuracy for filename
        acc_str = f"acc{acc1:.4f}"
        
        # Create checkpoint data
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
            "config": config_manager.config,
        }
        
        # Add accuracy metrics
        checkpoint_data["valid_acc"] = acc1
        if acc5 is not None:
            checkpoint_data["valid_acc5"] = acc5
        if acc10 is not None:
            checkpoint_data["valid_acc10"] = acc10
        
        # Save the checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Log the save action
        if writer is not None:
            checkpoint_type = "best" if is_best else "last" if is_last else "periodic"
            writer.add_text(
                "checkpoint",
                f"Saved {checkpoint_type} model at epoch {epoch} with top1: {acc1:.4f}" +
                (f", top5: {acc5:.4f}" if acc5 is not None else "") +
                (f", top10: {acc10:.4f}" if acc10 is not None else ""),
                epoch,
            )
    
    # Reset metrics
    accuracy_top1.reset()
    accuracy_top5.reset()
    accuracy_top10.reset()
    
    # Log the adapter output shape for debugging
    sample_input = torch.randn(1, 3, 32, 32).to(device)
    adapter_out = model[0](sample_input)
    print("Adapter Output Type:", type(adapter_out))
    print("Adapter Output Shape(s):", [x.shape for x in adapter_out] if isinstance(adapter_out, (list, tuple)) else adapter_out.shape)
    
    def train(dataset, log_prefix, epoch):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        total_samples = 0
        step = epoch * len(dataset)
        
        # Reset metrics
        accuracy_top1.reset()
    
        # Log the full LR schedule for the first epoch to visualize it
        if epoch == 0 and writer is not None:
            # Get the base LR
            base_lr = training_config.get('lr', 5e-4)
            # Log the theoretical LR curve for visualization
            total_steps = training_config.get('epochs', 100) * len(dataset)
            
            # Create a temporary scheduler just for visualization
            import torch.optim as optim
            dummy_model = torch.nn.Linear(1, 1)
            dummy_opt = optim.SGD(dummy_model.parameters(), lr=base_lr)
            
            # Create the appropriate scheduler based on the config
            if scheduler_type == "step":
                vis_scheduler = WarmupStepLR(
                    dummy_opt,
                    warmup_steps=warmup_steps,
                    milestones=scheduler_config.get('step_milestones', [15000, 30000]),
                    gamma=scheduler_config.get('step_gamma', 0.5)
                )
            else:  # Default to cosine scheduler
                vis_scheduler = ImprovedCosineWithWarmupLR(
                    dummy_opt,
                    training_steps=total_steps,
                    warmup_steps=warmup_steps,
                    plateau_steps=scheduler_config.get('plateau_steps', 2000),
                    min_lr_ratio=scheduler_config.get('min_lr_ratio', 0.1),
                    restart_steps=scheduler_config.get('restart_steps', 0)
                )
            
            # Update the text annotations for the scheduler
            writer.add_text("lr_schedule/type", scheduler_type, 0)
            writer.add_text("lr_schedule/base_lr", f"{base_lr}", 0)
            writer.add_text("lr_schedule/warmup_steps", f"{warmup_steps}", 0)
            
            if scheduler_type == "step":
                writer.add_text("lr_schedule/step_milestones", f"{scheduler_config.get('step_milestones', [15000, 30000])}", 0)
                writer.add_text("lr_schedule/step_gamma", f"{scheduler_config.get('step_gamma', 0.5)}", 0)
            else:
                writer.add_text("lr_schedule/plateau_steps", f"{scheduler_config.get('plateau_steps', 2000)}", 0)
                writer.add_text("lr_schedule/min_lr", f"{base_lr * scheduler_config.get('min_lr_ratio', 0.1)}", 0)
                if scheduler_config.get('restart_steps', 0) > 0:
                    writer.add_text("lr_schedule/restart_steps", f"{scheduler_config.get('restart_steps', 0)}", 0)
    
            # Sample the LR curve (up to 1000 points)
            sample_rate = max(1, total_steps // 1000)
            for i in range(total_steps):
                if i % sample_rate == 0:
                    writer.add_scalar("train/lr_schedule", vis_scheduler.get_last_lr()[0], i)
                vis_scheduler.step()
    
        with tqdm(dataset) as pbar:
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(device), targets.to(device)
                batch_size = inputs.size(0)
                total_samples += batch_size
    
                optimizer.zero_grad()
    
                with torch.autocast(device.type):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
    
                loss.backward()
    
                optimizer.step()
                scheduler.step()
    
                # Calculate accuracies
                acc = accuracy_top1(outputs, targets)
                lr = scheduler.get_last_lr()[0]
    
                # Update running statistics
                running_loss += loss.item() * batch_size
                running_acc += acc * batch_size
    
                pbar.set_description(
                    f"{log_prefix} | loss: {loss.item():.3f}, acc: {100.0 * acc:.3f}, lr: {lr:.3e}"
                )
    
                # Log to TensorBoard (per step)
                if writer is not None:
                    writer.add_scalar("train/step_loss", loss.item(), step + batch_idx)
                    writer.add_scalar("train/step_accuracy", acc, step + batch_idx)
                    writer.add_scalar("train/learning_rate", lr, step + batch_idx)
    
        # Compute final metrics
        final_acc = accuracy_top1.compute()
        accuracy_top1.reset()
    
        # Log epoch averages to TensorBoard
        if writer is not None:
            writer.add_scalar("train/epoch_loss", running_loss / total_samples, epoch)
            writer.add_scalar("train/epoch_accuracy", final_acc, epoch)
    
    
    @torch.no_grad()
    def evaluate(dataset, log_prefix="VALID", epoch=0, is_test=False):
        model.eval()
    
        loss = 0
        total = 0
        
        # Reset metrics
        accuracy_top1.reset()
        accuracy_top5.reset()
        accuracy_top10.reset()
    
        with torch.autocast(device.type):
            for inputs, targets in dataset:
                inputs, targets = inputs.to(device), targets.to(device)
                batch_size = inputs.size(0)
                total += batch_size
    
                outputs = model(inputs)
                loss += criterion(outputs, targets).item() * batch_size
                
                # Update accuracy metrics
                accuracy_top1.update(outputs, targets)
                accuracy_top5.update(outputs, targets)
                accuracy_top10.update(outputs, targets)
    
            # Compute final metrics
            acc1 = accuracy_top1.compute()
            acc5 = accuracy_top5.compute()
            acc10 = accuracy_top10.compute()
            avg_loss = loss / total
            
            # Reset metrics
            accuracy_top1.reset()
            accuracy_top5.reset()
            accuracy_top10.reset()
            
            # Print results
            print(f"{log_prefix} | loss: {avg_loss:.3f}, top1: {100.0 * acc1:.3f}%, top5: {100.0 * acc5:.3f}%, top10: {100.0 * acc10:.3f}%")
    
            # Log to TensorBoard
            if writer is not None:
                prefix = "test" if is_test else "valid"
                writer.add_scalar(f"{prefix}/loss", avg_loss, epoch)
                writer.add_scalar(f"{prefix}/top1_accuracy", acc1, epoch)
                writer.add_scalar(f"{prefix}/top5_accuracy", acc5, epoch)
                writer.add_scalar(f"{prefix}/top10_accuracy", acc10, epoch)
    
        return avg_loss, acc1, acc5, acc10
    
    
    # Track best model
    best_valid_acc = 0.0
    best_epoch = 0
    
    # Training loop
    num_epochs = training_config.get('epochs', 100)
    # Save checkpoint every N epochs
    checkpoint_interval = training_config.get('checkpoint_interval', 10)
    
    for epoch in range(num_epochs):
        train(train_batches, f"TRAIN | EPOCH {epoch}", epoch)
        valid_loss, valid_acc1, valid_acc5, valid_acc10 = evaluate(valid_batches, f"VALID | EPOCH {epoch}", epoch)
        
        # Save periodic checkpoint if the interval is reached
        if writer is not None and checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            run_name = os.path.basename(writer.log_dir)
            
            # Save periodic checkpoint with detailed filename
            periodic_checkpoint_path = os.path.join(
                writer.log_dir, 
                f"checkpoint_{run_name}_epoch{epoch}_acc{valid_acc1:.4f}.pt"
            )
            
            # Save the checkpoint with detailed metrics
            save_checkpoint(
                periodic_checkpoint_path, 
                epoch, 
                valid_acc1, 
                valid_acc5, 
                valid_acc10,
                is_periodic=True
            )
    
        # Save best model checkpoint (based on top1 accuracy)
        if valid_acc1 > best_valid_acc:
            best_valid_acc = valid_acc1
            best_epoch = epoch
            if writer is not None:
                # Get run name from the log path (last part of the path)
                run_name = os.path.basename(writer.log_dir)
                
                # Save with detailed filename including accuracy
                checkpoint_path = os.path.join(
                    writer.log_dir, 
                    f"best_model_{run_name}_epoch{epoch}_acc{valid_acc1:.4f}.pt"
                )
                
                # Save the checkpoint with detailed metrics
                save_checkpoint(
                    checkpoint_path, 
                    epoch, 
                    valid_acc1, 
                    valid_acc5, 
                    valid_acc10,
                    is_best=True
                )
                
                # Also save with a consistent name for easy loading
                best_model_path = os.path.join(writer.log_dir, "best_model.pt")
                # Remove previous file if it exists
                if os.path.exists(best_model_path):
                    os.remove(best_model_path)
                
                # Save with the standard name
                save_checkpoint(
                    best_model_path, 
                    epoch, 
                    valid_acc1, 
                    valid_acc5, 
                    valid_acc10
                )
                
                # Log model architecture and parameter count
                writer.add_text("model/architecture", str(model), 0)
                writer.add_scalar("model/parameter_count", compute_metrics["params"], 0)
    
    # Final evaluation on test set
    test_loss, test_acc1, test_acc5, test_acc10 = evaluate(test_batches, f"TEST FINAL", num_epochs-1, is_test=True)
    print(f"\nFinal Test Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Top-1 Accuracy: {100.0 * test_acc1:.2f}%")
    print(f"Top-5 Accuracy: {100.0 * test_acc5:.2f}%")
    print(f"Top-10 Accuracy: {100.0 * test_acc10:.2f}%")
    
    # Save last checkpoint
    if writer is not None:
        # Get run name from the log path (last part of the path)
        run_name = os.path.basename(writer.log_dir)
        
        # Save with detailed filename including accuracy
        last_checkpoint_path = os.path.join(
            writer.log_dir, 
            f"last_model_{run_name}_epoch{num_epochs-1}_acc{test_acc1:.4f}.pt"
        )
        
        # Save the checkpoint with detailed metrics
        save_checkpoint(
            last_checkpoint_path, 
            num_epochs - 1, 
            test_acc1, 
            test_acc5, 
            test_acc10,
            is_last=True
        )
        
        # Also save with a consistent name for easy loading
        last_model_path = os.path.join(writer.log_dir, "last_model.pt")
        # Remove previous file if it exists
        if os.path.exists(last_model_path):
            os.remove(last_model_path)
            
        # Save with the standard name
        save_checkpoint(
            last_model_path, 
            num_epochs - 1, 
            test_acc1, 
            test_acc5, 
            test_acc10
        )
    
    if writer is not None:
        # Get and add scheduler information
        scheduler_info = scheduler.get_scheduler_type()
        writer.add_text("scheduler/type", scheduler_info["type"], 0)
        for key, value in scheduler_info.items():
            if key != "type":
                writer.add_text(f"scheduler/{key}", str(value), 0)
                
        # Log compute metrics
        writer.add_scalar("model/flops", compute_metrics["flops"], 0)
        writer.add_scalar("model/flops_G", compute_metrics["flops_G"], 0)
        writer.add_scalar("model/flops_per_param", compute_metrics["flops_per_param"], 0)
        writer.add_text("model/compute_efficiency", f"FLOPs: {compute_metrics['flops']:,} ({compute_metrics['flops_G']:.2f} GFLOPS)", 0)
        writer.add_text("model/compute_efficiency", f"FLOPs per parameter: {compute_metrics['flops_per_param']:.2f}", 0)
        
        # Add model hyperparameters to logging
        # Convert the flat config to standard format for hparams
        flat_config = config_manager.get_flat_config()
        writer.add_hparams(
            flat_config,  # Log the entire flattened config as hyperparameters
            {
                "hparam/test_loss": test_loss,
                "hparam/test_top1_accuracy": test_acc1,
                "hparam/test_top5_accuracy": test_acc5,
                "hparam/test_top10_accuracy": test_acc10,
                "hparam/best_valid_top1_accuracy": best_valid_acc,
                "hparam/best_epoch": best_epoch,
                "hparam/model_flops_G": compute_metrics["flops_G"],
            },
        )
        
        # Log augmentation information specifically
        if augmentation_config.get('use_randaug', False):
            writer.add_text("augmentation/type", "RandAugment", 0)
            writer.add_text("augmentation/randaug_n", str(augmentation_config.get('randaug_n', 2)), 0)
            writer.add_text("augmentation/randaug_m", str(augmentation_config.get('randaug_m', 9)), 0)
        else:
            writer.add_text("augmentation/type", "Standard (RandomCrop+HorizontalFlip)", 0)
        
        writer.close()

if __name__ == "__main__":
    main()
