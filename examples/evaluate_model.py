import sys
sys.path.append("..")
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import torchmetrics
from flash_perceiver.utils.config import ConfigManager
import os
import yaml

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate a trained Perceiver model on CIFAR-100")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to model checkpoint file")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for evaluation")
    parser.add_argument("--data_root", type=str, default="../data",
                        help="Path to data directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for evaluation")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Extract config from checkpoint
    if "config" in checkpoint:
        config = checkpoint["config"]
        print("Using configuration from checkpoint")
    else:
        print("No configuration found in checkpoint, using default")
        config = {}
    
    # Convert config dict to ConfigManager if needed
    if isinstance(config, dict):
        # Save config to a temporary file
        temp_config_path = "temp_config.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        config_manager = ConfigManager(temp_config_path)
        os.remove(temp_config_path)  # Clean up
    else:
        config_manager = config
    
    # Get model configuration
    model_config = config_manager.get_model_config() if hasattr(config_manager, "get_model_config") else {}
    
    # Print model information
    print(f"Model was trained for {checkpoint.get('epoch', 'unknown')} epochs")
    print(f"Validation accuracy: {checkpoint.get('valid_acc', 0.0)*100:.2f}%")
    if "valid_acc5" in checkpoint:
        print(f"Top-5 validation accuracy: {checkpoint.get('valid_acc5', 0.0)*100:.2f}%")
    if "valid_acc10" in checkpoint:
        print(f"Top-10 validation accuracy: {checkpoint.get('valid_acc10', 0.0)*100:.2f}%")
    
    # Prepare test dataset
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_set = CIFAR100(args.data_root, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize accuracy metrics
    device = torch.device(args.device)
    accuracy_top1 = torchmetrics.Accuracy(task="multiclass", num_classes=100, top_k=1).to(device)
    accuracy_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=100, top_k=5).to(device)
    accuracy_top10 = torchmetrics.Accuracy(task="multiclass", num_classes=100, top_k=10).to(device)
    
    # Load model from checkpoint
    from flash_perceiver.perceiver import Perceiver
    from flash_perceiver.adapters import ImageAdapter
    from flash_perceiver.utils.encodings import PerceiverPositionalEncoding
    
    # Extract positional encoding config
    pos_encoding_config = config_manager.get_pos_encoding_config() if hasattr(config_manager, "get_pos_encoding_config") else {}
    
    # Recreate the model architecture
    model = nn.Sequential(
        ImageAdapter(
            embed_dim=None,
            pos_encoding=PerceiverPositionalEncoding(
                pos_encoding_config.get("dims", 2), 
                n_bands=pos_encoding_config.get("bands", 8), 
                max_res=pos_encoding_config.get("max_res", 32), 
                concat_pos=True
            ),
        ),
        Perceiver(
            input_dim=model_config.get("input_dim", 37),
            depth=model_config.get("depth", 1),
            output_dim=model_config.get("output_dim", 100),
            num_latents=model_config.get("num_latents", 512),
            latent_dim=model_config.get("latent_dim", 256),
            cross_heads=model_config.get("cross_heads", 1),
            cross_head_dim=model_config.get("cross_head_dim", 64),
            cross_rotary_emb_dim=model_config.get("cross_rotary_emb_dim", 0),
            cross_attn_dropout=model_config.get("cross_attn_dropout", 0.0),
            latent_heads=model_config.get("latent_heads", 8),
            latent_head_dim=model_config.get("latent_head_dim", 64),
            latent_rotary_emb_dim=model_config.get("latent_rotary_emb_dim", 0),
            latent_attn_dropout=model_config.get("latent_attn_dropout", 0.0),
            latent_drop=model_config.get("latent_drop", 0.0),
            weight_tie_layers=model_config.get("weight_tie_layers", False),
            gated_mlp=model_config.get("gated_mlp", True),
            self_per_cross_attn=model_config.get("self_per_cross_attn", 1),
            output_mode=model_config.get("output_mode", "average"),
            num_zero_tokens=model_config.get("num_zero_tokens", None),
            use_flash_attn=model_config.get("use_flash_attn", True),
        ),
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {param_count:,} trainable parameters")
    
    # Evaluate model
    print("\nEvaluating model on CIFAR-100 test set...")
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * batch_size
            
            # Update accuracy metrics
            accuracy_top1.update(outputs, targets)
            accuracy_top5.update(outputs, targets)
            accuracy_top10.update(outputs, targets)
    
    # Compute final metrics
    avg_loss = total_loss / total_samples
    acc1 = accuracy_top1.compute()
    acc5 = accuracy_top5.compute()
    acc10 = accuracy_top10.compute()
    
    # Report results
    print(f"\nTest results:")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Top-1 Accuracy: {100.0 * acc1:.2f}%")
    print(f"Top-5 Accuracy: {100.0 * acc5:.2f}%")
    print(f"Top-10 Accuracy: {100.0 * acc10:.2f}%")

if __name__ == "__main__":
    main() 