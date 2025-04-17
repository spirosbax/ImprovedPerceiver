import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class WarmupStepLR(LambdaLR):
    """
    Learning rate scheduler with linear warmup and step-based LR reduction.
    The LR will linearly increase during warmup_steps, then stay at peak value
    until reaching the specified milestone epochs where it will be multiplied
    by gamma (typically 0.5 or 0.1) at each milestone.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        milestones: list,  # List of steps at which to reduce LR
        gamma: float = 0.5,  # Multiplicative factor for LR reduction
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.milestones = sorted(milestones)
        self.gamma = gamma

        def lr_lambda(current_step):
            # Warmup phase
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            # Calculate step reduction based on milestones
            # Each time we pass a milestone, gamma is applied
            reduction_factor = self.gamma ** sum(current_step >= milestone for milestone in self.milestones)
            return reduction_factor

        super().__init__(optimizer, lr_lambda, last_epoch)

    def get_scheduler_type(self):
        """Return a dictionary with scheduler type and parameters for logging"""
        return {
            "type": "WarmupStepLR",
            "warmup_steps": self.warmup_steps,
            "milestones": self.milestones,
            "gamma": self.gamma
        }


class SimpleCosineWithWarmupLR(LambdaLR):
    """
    Simple cosine learning rate scheduler with warmup and min_lr.
    The LR will linearly increase during warmup_steps, then follow a cosine decay
    to min_lr over the remaining training steps.
    """
    def __init__(
        self, 
        optimizer: Optimizer, 
        training_steps: int, 
        warmup_steps: int = 0,
        min_lr: float = 0.0,  # Absolute minimum LR value
        last_epoch: int = -1,
    ):
        self.training_steps = training_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        
        # Store base_lrs to calculate absolute min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        def lr_lambda(current_step):
            # Warmup phase
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            # Cosine decay phase
            progress = float(current_step - warmup_steps) / float(max(1, training_steps - warmup_steps))
            # Basic cosine decay from 1.0 to 0.0
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            
            # Scale between 1.0 and min_lr/base_lr
            if self.min_lr > 0:
                # Calculate min_lr as a ratio of base_lr for the current param group
                idx = self._step_count % len(self.base_lrs)
                base_lr = self.base_lrs[idx]
                min_lr_ratio = self.min_lr / base_lr
                # Ensure we don't go below min_lr
                return max(min_lr_ratio, cosine_decay)
            else:
                return cosine_decay

        super().__init__(optimizer, lr_lambda, last_epoch)
        
    def get_scheduler_type(self):
        """Return a dictionary with scheduler type and parameters for logging"""
        return {
            "type": "SimpleCosineWithWarmupLR",
            "training_steps": self.training_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr
        }


class ImprovedCosineWithWarmupLR(LambdaLR):
    def __init__(
        self, 
        optimizer: Optimizer, 
        training_steps: int, 
        warmup_steps: int = 0, 
        plateau_steps: int = 0,  # New parameter for plateau period
        min_lr_ratio: float = 0.0,  # New parameter for minimum LR as a ratio of base LR
        num_cycles: float = 0.5,  # Can be > 1 for multiple restarts
        restart_steps: int = 0,  # New parameter for periodic restarts (0 means no restarts)
    ):
        self.training_steps = training_steps
        self.warmup_steps = warmup_steps
        self.plateau_steps = plateau_steps
        self.min_lr_ratio = min_lr_ratio
        self.num_cycles = num_cycles
        self.restart_steps = restart_steps

        def lr_lambda(current_step):
            # Warmup phase
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            # Plateau phase (maintain peak LR)
            if current_step < warmup_steps + plateau_steps:
                return 1.0
            
            # Decay phase with optional restarts
            decay_steps = training_steps - warmup_steps - plateau_steps
            current_decay_step = current_step - warmup_steps - plateau_steps
            
            # Handle periodic restarts if enabled
            if restart_steps > 0 and decay_steps > 0:
                # Calculate which restart cycle we're in
                cycle = math.floor(current_decay_step / restart_steps)
                # Get the step within the current cycle
                cycle_step = current_decay_step - cycle * restart_steps
                # Progress within the current cycle
                cycle_progress = cycle_step / restart_steps
                # The cosine decay for this cycle
                cosine_value = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
                # Scale between 1.0 and min_lr_ratio
                return max(min_lr_ratio, cosine_value)
            else:
                # Standard cosine decay without restarts
                progress = float(current_decay_step) / float(max(1, decay_steps))
                # Ensure we don't go below min_lr_ratio
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
                return max(min_lr_ratio, cosine_decay)

        super().__init__(optimizer, lr_lambda, -1)
        
    def get_scheduler_type(self):
        """Return a dictionary with scheduler type and parameters for logging"""
        return {
            "type": "ImprovedCosineWithWarmupLR",
            "training_steps": self.training_steps,
            "warmup_steps": self.warmup_steps,
            "plateau_steps": self.plateau_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "num_cycles": self.num_cycles,
            "restart_steps": self.restart_steps
        }


# Keep the original for backward compatibility
class CosineWithWarmupLR(LambdaLR):
    def __init__(self, optimizer: Optimizer, training_steps: int, warmup_steps: int = 0, num_cycles: float = 0.5):
        self.training_steps = training_steps
        self.warmup_steps = warmup_steps
        self.num_cycles = num_cycles
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, training_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        super().__init__(optimizer, lr_lambda, -1)
        
    def get_scheduler_type(self):
        """Return a dictionary with scheduler type and parameters for logging"""
        return {
            "type": "CosineWithWarmupLR",
            "training_steps": self.training_steps,
            "warmup_steps": self.warmup_steps,
            "num_cycles": self.num_cycles
        }
