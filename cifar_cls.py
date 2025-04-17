import sys
import argparse
import torch
import os
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from flash_perceiver.utils.config import ConfigManager
from flash_perceiver.perceiver_module import CIFARPerceiverModule
from flash_perceiver.data_modules import CIFARDataModule

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="CIFAR100 Training with Perceiver (Lightning)")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to YAML config file")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    
    # Extract configurations
    training_config = config_manager.get_training_config()
    logging_config = config_manager.get_logging_config()
    
    # Set device
    device = config_manager.get_device()
    
    # Initialize TensorBoard logger if enabled
    logger = None
    if logging_config.get('use_tensorboard', True):
        log_path = os.path.join(
            logging_config.get('logdir', './logs'), 
            f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        )
        logger = TensorBoardLogger(save_dir=log_path)
        print(f"TensorBoard logs will be saved to {log_path}")
        
        # Save the config to the log directory
        os.makedirs(log_path, exist_ok=True)
        config_manager.save_config(os.path.join(log_path, "config.yaml"))
    
    # Create data module
    data_module = CIFARDataModule(config_manager)
    
    # Create model
    model = CIFARPerceiverModule(config_manager)
    
    # Setup callbacks
    callbacks = []
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc1',
        dirpath=log_path,
        filename='best_model_{epoch:02d}_{val_acc1:.4f}',
        save_top_k=5,
        mode='max',
        save_last=True,
        every_n_epochs=training_config.get('checkpoint_interval', 10),
    )
    callbacks.append(checkpoint_callback)
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=training_config.get('epochs', 100),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=logger,
        callbacks=callbacks,
        accumulate_grad_batches=training_config.get('gradient_accumulation_steps', 1),
        precision=16 if device.type == 'cuda' else 32,  # Use 16-bit precision on GPU
        log_every_n_steps=50,
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    trainer.test(model, data_module)

if __name__ == "__main__":
    main() 