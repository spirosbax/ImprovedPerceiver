import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

class SimpleCosineWithWarmupLR(_LRScheduler):
    """
    PyTorch Lightning compatible scheduler that does linear warmup followed by cosine decay.
    Linearly increases learning rate from 0 to max_lr over `warmup_steps` training steps.
    Decreases learning rate from max_lr to min_lr over remaining `training_steps - warmup_steps` steps following a cosine curve.
    """
    def __init__(self, optimizer, training_steps, warmup_steps, min_lr=0.0, last_epoch=-1):
        """
        Args:
            optimizer: Wrapped optimizer.
            training_steps: Total number of training steps.
            warmup_steps: Number of steps for the warmup phase.
            min_lr: Minimum learning rate at the end of training. Default: 0.
            last_epoch: The index of the last epoch. Default: -1.
        """
        self.training_steps = training_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            # Just return the base LR values when users call get_lr() directly
            return [base_lr for base_lr in self.base_lrs]
        
        step = self.last_epoch

        # Get base learning rate
        if step < self.warmup_steps:
            # Linear warmup
            lr_factor = float(step) / float(max(1, self.warmup_steps))
            return [base_lr * lr_factor for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = float(step - self.warmup_steps) / float(max(1, self.training_steps - self.warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            
            # Map cosine decay from [0, 1] to [min_lr, base_lr]
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]
    
    def get_scheduler_type(self):
        """Return information about the scheduler configuration."""
        return {
            "type": "cosine_with_warmup",
            "warmup_steps": self.warmup_steps,
            "training_steps": self.training_steps,
            "min_lr": self.min_lr
        }

    def state_dict(self):
        state = super().state_dict()
        state['training_steps'] = self.training_steps
        state['warmup_steps'] = self.warmup_steps
        state['min_lr'] = self.min_lr
        return state

    def load_state_dict(self, state_dict):
        self.training_steps = state_dict.pop('training_steps')
        self.warmup_steps = state_dict.pop('warmup_steps')
        self.min_lr = state_dict.pop('min_lr')
        super().load_state_dict(state_dict)


class WarmupStepLR(_LRScheduler):
    """
    PyTorch Lightning compatible scheduler that implements a warmup period followed by step decay.
    The learning rate is linearly increased from 0 to the initial lr over `warmup_steps`,
    then decays by a factor of gamma at each milestone.
    """
    def __init__(self, optimizer, warmup_steps, milestones, gamma=0.1, last_epoch=-1):
        """
        Args:
            optimizer: Wrapped optimizer.
            warmup_steps: Number of steps for the warmup phase.
            milestones: List of step indices at which the learning rate will be decreased.
            gamma: Multiplicative factor of learning rate decay. Default: 0.1.
            last_epoch: The index of the last epoch. Default: -1.
        """
        self.warmup_steps = warmup_steps
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            # Just return the base LR values when users call get_lr() directly
            return [base_lr for base_lr in self.base_lrs]
        
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Linear warmup
            lr_factor = float(step) / float(max(1, self.warmup_steps))
            return [base_lr * lr_factor for base_lr in self.base_lrs]
        else:
            # Step decay
            # Calculate decay factor based on how many milestones we've passed
            decay_factor = self.gamma ** len([m for m in self.milestones if m <= step])
            return [base_lr * decay_factor for base_lr in self.base_lrs]
    
    def get_scheduler_type(self):
        """Return information about the scheduler configuration."""
        return {
            "type": "step_with_warmup",
            "warmup_steps": self.warmup_steps,
            "milestones": self.milestones,
            "gamma": self.gamma
        }

    def state_dict(self):
        state = super().state_dict()
        state['warmup_steps'] = self.warmup_steps
        state['milestones'] = self.milestones
        state['gamma'] = self.gamma
        return state

    def load_state_dict(self, state_dict):
        self.warmup_steps = state_dict.pop('warmup_steps')
        self.milestones = state_dict.pop('milestones')
        self.gamma = state_dict.pop('gamma')
        super().load_state_dict(state_dict) 