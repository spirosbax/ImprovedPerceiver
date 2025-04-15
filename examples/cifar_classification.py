import sys
sys.path.append("..")
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from pytorch_lamb import Lamb
from tqdm import tqdm
from flash_perceiver import utils #, Perceiver
from flash_perceiver.perceiver import Perceiver
from flash_perceiver.adapters import ImageAdapter
from flash_perceiver.utils.encodings import PerceiverPositionalEncoding
from flash_perceiver.utils.training import CosineWithWarmupLR
from torch.utils.tensorboard import SummaryWriter
import os
import json
from datetime import datetime
import torchmetrics

# Default configuration for the Perceiver model
DEFAULT_CONFIG = {
    # Model architecture
    "input_dim": 37,  # was 43
    "depth": 1,
    "output_dim": 10,
    "num_latents": 128,
    "latent_dim": 256,
    "cross_attn_dropout": 0.2,
    "latent_attn_dropout": 0.2,
    "self_per_cross_attn": 4,
    # Training parameters
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "warmup_steps": 1000,
    "batch_size": 512,
    "epochs": 100,
    # Positional encoding
    "pos_encoding_bands": 8,
    "pos_encoding_dims": 2,
    "pos_encoding_max_res": 32
}

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=DEFAULT_CONFIG["lr"], type=float, help="learning rate")
parser.add_argument("--device", default="cuda")
parser.add_argument("--data_root", default="../data")
parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
parser.add_argument("--use_tensorboard", action="store_true", help="Use TensorBoard for logging", default=True)
parser.add_argument("--logdir", default="./logs", help="TensorBoard log directory")
parser.add_argument("--config", type=str, help="Path to config JSON file to override defaults")
parser.add_argument("--num_latents", type=int, default=DEFAULT_CONFIG["num_latents"], help="Number of latent vectors")
parser.add_argument("--latent_dim", type=int, default=DEFAULT_CONFIG["latent_dim"], help="Dimension of latent vectors")
parser.add_argument("--cross_attn_dropout", type=float, default=DEFAULT_CONFIG["cross_attn_dropout"], help="Cross attention dropout rate")
parser.add_argument("--latent_attn_dropout", type=float, default=DEFAULT_CONFIG["latent_attn_dropout"], help="Latent attention dropout rate")
parser.add_argument("--self_per_cross_attn", type=int, default=DEFAULT_CONFIG["self_per_cross_attn"], help="Self attention blocks per cross attention")
args = parser.parse_args()

# Create a config by merging defaults with command line arguments and optional config file
config = DEFAULT_CONFIG.copy()

# Override with config file if provided
if args.config and os.path.exists(args.config):
    with open(args.config, 'r') as f:
        file_config = json.load(f)
        config.update(file_config)

# Override with command line arguments
config.update({
    "lr": args.lr,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "num_latents": args.num_latents,
    "latent_dim": args.latent_dim,
    "cross_attn_dropout": args.cross_attn_dropout,
    "latent_attn_dropout": args.latent_attn_dropout,
    "self_per_cross_attn": args.self_per_cross_attn,
})

# Initialize TensorBoard writer if enabled
writer = None
if args.use_tensorboard:
    log_path = os.path.join(
        args.logdir, f'cifar10_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    )
    writer = SummaryWriter(log_path)
    print(f"TensorBoard logs will be saved to {log_path}")
    
    # Log the configuration
    writer.add_text("config", json.dumps(config, indent=2), 0)

print("==> Preparing data..")

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

rng = torch.Generator(device=args.device)

train_set = CIFAR10(
    args.data_root, train=True, download=False, transform=transform_train
)
test_set = CIFAR10(args.data_root, train=False, download=False, transform=transform_test)

valid_set, train_set = random_split(train_set, [5000, 45000])

train_batches = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
valid_batches = DataLoader(valid_set, batch_size=config["batch_size"], shuffle=False)
test_batches = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)

print("==> Building model..")

model = nn.Sequential(
    ImageAdapter(
        embed_dim=None,
        pos_encoding=PerceiverPositionalEncoding(
            config["pos_encoding_dims"], 
            n_bands=config["pos_encoding_bands"], 
            max_res=config["pos_encoding_max_res"], 
            concat_pos=True
        ),
    ),
    Perceiver(
        input_dim=config["input_dim"],
        depth=config["depth"],
        output_dim=config["output_dim"],
        num_latents=config["num_latents"],
        latent_dim=config["latent_dim"],
        cross_attn_dropout=config["cross_attn_dropout"],
        latent_attn_dropout=config["latent_attn_dropout"],
        self_per_cross_attn=config["self_per_cross_attn"],
    ),
).to(args.device)

criterion = nn.CrossEntropyLoss()
optimizer = Lamb(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
scheduler = CosineWithWarmupLR(
    optimizer, training_steps=config["epochs"] * len(train_batches), warmup_steps=config["warmup_steps"]
)

# Store parameter count in config for logging
config["model_params"] = utils.numel(model)
print(f"Number of parameters: {config['model_params']:,}")

# Initialize metrics
accuracy_top1 = torchmetrics.Accuracy(task="multiclass", num_classes=10, top_k=1).to(args.device)
accuracy_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=10, top_k=5).to(args.device)
accuracy_top10 = torchmetrics.Accuracy(task="multiclass", num_classes=10, top_k=min(10, config["output_dim"])).to(args.device)

# Add sample input to TensorBoard graph
if writer is not None:
    sample_input = torch.randn(1, 3, 32, 32).to(args.device)
    writer.add_graph(model, sample_input)

sample_input = torch.randn(1, 3, 32, 32).to(args.device)
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

    with tqdm(dataset) as pbar:
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            batch_size = inputs.size(0)
            total_samples += batch_size

            optimizer.zero_grad()

            with torch.autocast(args.device):
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

    with torch.autocast(args.device):
        for inputs, targets in dataset:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
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

for epoch in range(config["epochs"]):
    train(train_batches, f"TRAIN | EPOCH {epoch}", epoch)
    valid_loss, valid_acc1, valid_acc5, valid_acc10 = evaluate(valid_batches, f"VALID | EPOCH {epoch}", epoch)

    # Save best model checkpoint (based on top1 accuracy)
    if valid_acc1 > best_valid_acc:
        best_valid_acc = valid_acc1
        best_epoch = epoch
        if writer is not None:
            checkpoint_path = os.path.join(args.logdir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "valid_acc": valid_acc1,
                    "valid_acc5": valid_acc5,
                    "valid_acc10": valid_acc10,
                    "config": config,  # Save config with the model
                },
                checkpoint_path,
            )
            writer.add_text(
                "checkpoint",
                f"Saved best model at epoch {epoch} with top1: {valid_acc1:.4f}, top5: {valid_acc5:.4f}, top10: {valid_acc10:.4f}",
                epoch,
            )
            # Log model architecture and parameter count
            writer.add_text("model/architecture", str(model), 0)
            writer.add_scalar("model/parameter_count", config["model_params"], 0)

# Final evaluation on test set
test_loss, test_acc1, test_acc5, test_acc10 = evaluate(test_batches, f"TEST FINAL", config["epochs"]-1, is_test=True)
print(f"\nFinal Test Results:")
print(f"Loss: {test_loss:.4f}")
print(f"Top-1 Accuracy: {100.0 * test_acc1:.2f}%")
print(f"Top-5 Accuracy: {100.0 * test_acc5:.2f}%")
print(f"Top-10 Accuracy: {100.0 * test_acc10:.2f}%")

# Save last checkpoint
if writer is not None:
    last_checkpoint_path = os.path.join(args.logdir, "last_model.pt")
    torch.save(
        {
            "epoch": config["epochs"] - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "test_acc": test_acc1,
            "test_acc5": test_acc5,
            "test_acc10": test_acc10,
            "config": config,  # Save config with the model
        },
        last_checkpoint_path,
    )
    writer.add_text(
        "checkpoint",
        f"Saved last model after {config['epochs']} epochs with test top1: {test_acc1:.4f}, top5: {test_acc5:.4f}, top10: {test_acc10:.4f}",
        config["epochs"] - 1,
    )

if writer is not None:
    # Add model hyperparameters to logging
    writer.add_hparams(
        config,  # Log the entire config as hyperparameters
        {
            "hparam/test_loss": test_loss,
            "hparam/test_top1_accuracy": test_acc1,
            "hparam/test_top5_accuracy": test_acc5,
            "hparam/test_top10_accuracy": test_acc10,
            "hparam/best_valid_top1_accuracy": best_valid_acc,
            "hparam/best_epoch": best_epoch,
        },
    )
    
    # Save the final config to a JSON file in the log directory
    config_path = os.path.join(log_path, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
        
    writer.close()
