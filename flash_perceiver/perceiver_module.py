import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from flash_perceiver.perceiver import Perceiver
from flash_perceiver.adapters import ImageAdapter
from flash_perceiver.utils.encodings import PerceiverPositionalEncoding
from flash_perceiver.utils.custom_schedulers import SimpleCosineWithWarmupLR, WarmupStepLR
from pytorch_lamb import Lamb

class CIFARPerceiverModule(pl.LightningModule):
    def __init__(self, config_manager):
        super().__init__()
        self.save_hyperparameters()
        
        # Store configuration
        self.config_manager = config_manager
        
        # Extract configurations
        self.model_config = config_manager.get_model_config()
        self.pos_encoding_config = config_manager.get_pos_encoding_config()
        self.training_config = config_manager.get_training_config()
        self.optimizer_config = config_manager.get_optimizer_config()
        self.scheduler_config = config_manager.get_scheduler_config()
        
        # Build model
        self.model = self._build_model()
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=100, top_k=1)
        self.val_acc1 = torchmetrics.Accuracy(task="multiclass", num_classes=100, top_k=1)
        self.val_acc5 = torchmetrics.Accuracy(task="multiclass", num_classes=100, top_k=5)
        self.val_acc10 = torchmetrics.Accuracy(task="multiclass", num_classes=100, top_k=10)
        self.test_acc1 = torchmetrics.Accuracy(task="multiclass", num_classes=100, top_k=1)
        self.test_acc5 = torchmetrics.Accuracy(task="multiclass", num_classes=100, top_k=5)
        self.test_acc10 = torchmetrics.Accuracy(task="multiclass", num_classes=100, top_k=10)
        
        # Calculate and log compute metrics
        self.compute_metrics = self._count_parameters_and_flops()
        
    def _build_model(self):
        """Build the Perceiver model for CIFAR100."""
        model = nn.Sequential(
            ImageAdapter(
                embed_dim=None,
                pos_encoding=PerceiverPositionalEncoding(
                    self.pos_encoding_config.get('dims', 2), 
                    n_bands=self.pos_encoding_config.get('bands', 8), 
                    max_res=self.pos_encoding_config.get('max_res', 32), 
                    concat_pos=True
                ),
            ),
            Perceiver(
                input_dim=self.model_config.get('input_dim', 37),
                depth=self.model_config.get('depth', 1),
                output_dim=self.model_config.get('output_dim', 100),
                num_latents=self.model_config.get('num_latents', 512),
                latent_dim=self.model_config.get('latent_dim', 256),
                cross_heads=self.model_config.get('cross_heads', 1),
                cross_head_dim=self.model_config.get('cross_head_dim', 64),
                cross_rotary_emb_dim=self.model_config.get('cross_rotary_emb_dim', 0),
                cross_attn_dropout=self.model_config.get('cross_attn_dropout', 0.0),
                latent_heads=self.model_config.get('latent_heads', 8),
                latent_head_dim=self.model_config.get('latent_head_dim', 64),
                latent_rotary_emb_dim=self.model_config.get('latent_rotary_emb_dim', 0),
                latent_attn_dropout=self.model_config.get('latent_attn_dropout', 0.0),
                latent_drop=self.model_config.get('latent_drop', 0.0),
                weight_tie_layers=self.model_config.get('weight_tie_layers', False),
                gated_mlp=self.model_config.get('gated_mlp', True),
                self_per_cross_attn=self.model_config.get('self_per_cross_attn', 1),
                output_mode=self.model_config.get('output_mode', 'average'),
                num_zero_tokens=self.model_config.get('num_zero_tokens', None),
                use_flash_attn=self.model_config.get('use_flash_attn', True),
            ),
        )
        
        return model
    
    def _count_parameters_and_flops(self, input_shape=(1, 3, 32, 32)):
        """Count parameters and estimate FLOPs for the model"""
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Get model attributes for FLOP calculation
        if hasattr(self.model, 'num_latents') and hasattr(self.model, 'latent_dim'):
            # Direct access to model attributes
            L = self.model.num_latents
            D = self.model.latent_dim
        else:
            # Access through submodule (for Sequential model)
            for module in self.model.modules():
                if hasattr(module, 'num_latents') and hasattr(module, 'latent_dim'):
                    L = module.num_latents
                    D = module.latent_dim
                    break
            else:
                # Fallback to config values
                L = self.model_config.get('num_latents', 512)
                D = self.model_config.get('latent_dim', 256)
        
        # Image adapter converts 32x32x3 image to sequence of length N
        H, W = input_shape[2], input_shape[3]
        patch_size = 1  # Default assumption for Perceiver
        N = (H // patch_size) * (W // patch_size)
        
        # Estimate FLOPs for different components
        depth = self.model_config.get('depth', 1)
        self_per_cross_attn = self.model_config.get('self_per_cross_attn', 1)
        
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
        
        metrics = {
            "params": total_params,
            "flops": total_flops,
            "flops_G": total_flops / 1e9,
            "flops_per_param": total_flops / total_params
        }
        
        # Log compute metrics
        self.log_dict({
            "model/params": metrics["params"],
            "model/flops_G": metrics["flops_G"],
            "model/flops_per_param": metrics["flops_per_param"]
        })
        
        # Print compute metrics
        print(f"Number of parameters: {metrics['params']:,}")
        print(f"Estimated FLOPs: {metrics['flops']:,} ({metrics['flops_G']:.2f} GFLOPS)")
        print(f"FLOPs per parameter: {metrics['flops_per_param']:.2f}")
        
        return metrics

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        # Choose optimizer based on configuration
        optimizer_type = self.optimizer_config.get('type', 'lamb')
        learning_rate = self.training_config.get('lr', 5e-4)
        weight_decay = self.training_config.get('weight_decay', 1e-4)
        
        if optimizer_type.lower() == 'adamw':
            print(f"==> Using AdamW optimizer with lr={learning_rate}, weight_decay={weight_decay}")
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=self.optimizer_config.get('betas', (0.9, 0.999)),
                eps=self.optimizer_config.get('eps', 1e-8)
            )
        else:  # Default to Lamb
            print(f"==> Using Lamb optimizer with lr={learning_rate}, weight_decay={weight_decay}")
            optimizer = Lamb(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        
        # Configure learning rate scheduler
        scheduler_type = self.scheduler_config.get('type', 'cosine')
        
        # Calculate steps per epoch (accounting for gradient accumulation)
        steps_per_epoch = self.trainer.estimated_stepping_batches / self.trainer.max_epochs
        
        # Set warmup steps/epochs
        warmup_epochs = self.scheduler_config.get('warmup_epochs')
        if warmup_epochs is not None:
            # Convert epochs to steps
            warmup_steps = int(warmup_epochs * steps_per_epoch)
            print(f"Using warmup_epochs: {warmup_epochs} (converted to {warmup_steps} steps)")
        else:
            # Fallback to warmup_steps if provided, or default
            warmup_steps = self.scheduler_config.get('warmup_steps', 1000)
            warmup_epochs = warmup_steps / steps_per_epoch if steps_per_epoch > 0 else 0
            print(f"Using warmup_steps: {warmup_steps} (approximately {warmup_epochs:.2f} epochs)")
        
        if scheduler_type == "step":
            step_milestones = self.scheduler_config.get('step_milestones', [15000, 30000])
            step_gamma = self.scheduler_config.get('step_gamma', 0.5)
            
            print("\n==> Using step-based learning rate scheduler with:")
            print(f"    - {warmup_steps} warmup steps ({warmup_epochs:.2f} epochs)")
            print(f"    - LR reduction by factor of {step_gamma} at steps: {step_milestones}")
            print("")
            
            scheduler = {
                "scheduler": WarmupStepLR(
                    optimizer,
                    warmup_steps=warmup_steps,
                    milestones=step_milestones,
                    gamma=step_gamma
                ),
                "interval": "step",
                "frequency": 1,
                "name": "step_lr"
            }
        else:  # Default to cosine scheduler
            # Get min_lr parameter (absolute value)
            min_lr = self.scheduler_config.get('min_lr')
            if min_lr is None:
                # Backward compatibility with min_rl (previous typo)
                min_lr = self.scheduler_config.get('min_rl', 0.0)
            
            print("\n==> Using simple cosine learning rate scheduler with:")
            print(f"    - {warmup_steps} warmup steps ({warmup_epochs:.2f} epochs)")
            if min_lr > 0:
                print(f"    - Min LR: {min_lr:.6e}")
            else:
                print(f"    - Min LR: 0.0 (standard cosine decay to zero)")
            print("")
            
            # Calculate total training steps
            total_training_steps = self.trainer.estimated_stepping_batches
            
            scheduler = {
                "scheduler": SimpleCosineWithWarmupLR(
                    optimizer, 
                    training_steps=total_training_steps, 
                    warmup_steps=warmup_steps,
                    min_lr=min_lr
                ),
                "interval": "step",
                "frequency": 1,
                "name": "cosine_lr"
            }
        
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        
        # Update accuracy metric
        acc = self.train_acc(outputs, targets)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log learning rate
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        
        # Update accuracy metrics
        acc1 = self.val_acc1(outputs, targets)
        acc5 = self.val_acc5(outputs, targets)
        acc10 = self.val_acc10(outputs, targets)
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc1', acc1, on_epoch=True, prog_bar=True)
        self.log('val_acc5', acc5, on_epoch=True)
        self.log('val_acc10', acc10, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        
        # Update accuracy metrics
        acc1 = self.test_acc1(outputs, targets)
        acc5 = self.test_acc5(outputs, targets)
        acc10 = self.test_acc10(outputs, targets)
        
        # Log metrics
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc1', acc1, on_epoch=True)
        self.log('test_acc5', acc5, on_epoch=True)
        self.log('test_acc10', acc10, on_epoch=True)
        
        return loss
    
    def on_train_epoch_end(self):
        # Get epoch metrics
        train_acc = self.train_acc.compute()
        
        # Print epoch summary
        print(f"TRAIN | EPOCH {self.current_epoch} | loss: {self.trainer.callback_metrics['train_loss_epoch']:.3f}, acc: {train_acc:.3f}")
        
        # Reset metrics
        self.train_acc.reset()
    
    def on_validation_epoch_end(self):
        # Get epoch metrics
        val_acc1 = self.val_acc1.compute()
        val_acc5 = self.val_acc5.compute()
        val_acc10 = self.val_acc10.compute()
        
        # Print epoch summary
        print(f"VALID | EPOCH {self.current_epoch} | loss: {self.trainer.callback_metrics['val_loss']:.3f}, " 
              f"top1: {100.0 * val_acc1:.3f}%, top5: {100.0 * val_acc5:.3f}%, top10: {100.0 * val_acc10:.3f}%")
        
        # Reset metrics
        self.val_acc1.reset()
        self.val_acc5.reset()
        self.val_acc10.reset()
    
    def on_test_epoch_end(self):
        # Get test metrics
        test_acc1 = self.test_acc1.compute()
        test_acc5 = self.test_acc5.compute()
        test_acc10 = self.test_acc10.compute()
        
        # Print test summary
        print(f"\nFinal Test Results:")
        print(f"Loss: {self.trainer.callback_metrics['test_loss']:.4f}")
        print(f"Top-1 Accuracy: {100.0 * test_acc1:.2f}%")
        print(f"Top-5 Accuracy: {100.0 * test_acc5:.2f}%")
        print(f"Top-10 Accuracy: {100.0 * test_acc10:.2f}%")
        
        # Reset metrics
        self.test_acc1.reset()
        self.test_acc5.reset()
        self.test_acc10.reset() 