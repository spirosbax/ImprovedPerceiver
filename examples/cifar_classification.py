import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from pytorch_lamb import Lamb
from tqdm import tqdm
from flash_perceiver import utils, Perceiver
from flash_perceiver.adapters import ImageAdapter
from flash_perceiver.utils.encodings import NeRFPositionalEncoding
from flash_perceiver.utils.training import CosineWithWarmupLR
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime


parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=5e-4, type=float, help="learning rate")
parser.add_argument("--device", default="cuda")
parser.add_argument("--data_root", default="./data")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument(
    "--use_tensorboard",
    action="store_true",
    help="Use TensorBoard for logging",
)
parser.add_argument("--logdir", default="./runs", help="TensorBoard log directory")
args = parser.parse_args()

# Initialize TensorBoard writer if enabled
writer = None
if args.use_tensorboard:
    log_path = os.path.join(
        args.logdir, f'cifar10_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    )
    writer = SummaryWriter(log_path)
    print(f"TensorBoard logs will be saved to {log_path}")

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
    args.data_root, train=True, download=True, transform=transform_train
)
test_set = CIFAR10(args.data_root, train=False, download=True, transform=transform_test)

valid_set, train_set = random_split(train_set, [5000, 45000])

train_batches = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
valid_batches = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
test_batches = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

print("==> Building model..")

model = nn.Sequential(
    ImageAdapter(
        embed_dim=64,
        pos_encoding=NeRFPositionalEncoding(2),
    ),
    Perceiver(
        input_dim=64,
        depth=1,
        output_dim=10,
        num_latents=128,
        latent_dim=256,
        cross_attn_dropout=0.2,
        latent_attn_dropout=0.2,
        self_per_cross_attn=4,
    ),
).to(args.device)

criterion = nn.CrossEntropyLoss()
optimizer = Lamb(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = CosineWithWarmupLR(
    optimizer, training_steps=args.epochs * len(train_batches), warmup_steps=1000
)

print(f"Number of parameters: {utils.numel(model):,}")

# Add sample input to TensorBoard graph
if writer is not None:
    sample_input = torch.randn(1, 3, 32, 32).to(args.device)
    writer.add_graph(model, sample_input)


def train(dataset, log_prefix, epoch):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0
    step = epoch * len(dataset)

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

            acc = outputs.argmax(-1).eq(targets).float().mean().item()
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

    # Log epoch averages to TensorBoard
    if writer is not None:
        writer.add_scalar("train/epoch_loss", running_loss / total_samples, epoch)
        writer.add_scalar("train/epoch_accuracy", running_acc / total_samples, epoch)


@torch.no_grad()
def evaluate(dataset, log_prefix="VALID", epoch=0, is_test=False):
    model.eval()

    loss = 0
    correct = 0
    total = 0

    with torch.autocast(args.device):
        for inputs, targets in dataset:
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            outputs = model(inputs)

            loss += criterion(outputs, targets).item() * targets.size(0)
            total += targets.size(0)
            correct += outputs.argmax(-1).eq(targets).sum().item()

        avg_loss = loss / total
        accuracy = correct / total
        print(f"{log_prefix} | loss: {avg_loss:.3f}, acc: {100.0 * accuracy:.3f}")

        # Log to TensorBoard
        if writer is not None:
            prefix = "test" if is_test else "valid"
            writer.add_scalar(f"{prefix}/loss", avg_loss, epoch)
            writer.add_scalar(f"{prefix}/accuracy", accuracy, epoch)

    return avg_loss, accuracy


# Track best model
best_valid_acc = 0.0
best_epoch = 0

for epoch in range(args.epochs):
    train(train_batches, f"TRAIN | EPOCH {epoch}", epoch)
    valid_loss, valid_acc = evaluate(valid_batches, f"VALID | EPOCH {epoch}", epoch)

    # Save best model checkpoint
    if valid_acc > best_valid_acc and writer is not None:
        best_valid_acc = valid_acc
        best_epoch = epoch
        checkpoint_path = os.path.join(args.logdir, "best_model.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "valid_acc": valid_acc,
            },
            checkpoint_path,
        )
        if writer is not None:
            writer.add_text(
                "checkpoint",
                f"Saved best model at epoch {epoch} with accuracy {valid_acc:.4f}",
                epoch,
            )

test_loss, test_acc = evaluate(test_batches, f"TEST", epoch, is_test=True)

if writer is not None:
    writer.add_hparams(
        {"lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs},
        {
            "hparam/test_loss": test_loss,
            "hparam/test_accuracy": test_acc,
            "hparam/best_valid_accuracy": best_valid_acc,
            "hparam/best_epoch": best_epoch,
        },
    )
    writer.close()
