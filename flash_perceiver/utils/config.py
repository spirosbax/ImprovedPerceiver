import os
import yaml
from typing import Dict, Any, Optional
import torch

class ConfigManager:
    """
    Configuration manager for the flash_perceiver project.
    Handles loading and accessing configuration from YAML files.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load the configuration from the YAML file.
        
        Returns:
            Dict: Configuration dictionary
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return config
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get the model configuration.
        
        Returns:
            Dict: Model configuration
        """
        return self.config.get('model', {})
    
    def get_pos_encoding_config(self) -> Dict[str, Any]:
        """
        Get the positional encoding configuration.
        
        Returns:
            Dict: Positional encoding configuration
        """
        return self.config.get('pos_encoding', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """
        Get the training configuration.
        
        Returns:
            Dict: Training configuration
        """
        return self.config.get('training', {})
    
    def get_augmentation_config(self) -> Dict[str, Any]:
        """
        Get the augmentation configuration.
        
        Returns:
            Dict: Augmentation configuration
        """
        return self.config.get('augmentation', {})
    
    def get_scheduler_config(self) -> Dict[str, Any]:
        """
        Get the scheduler configuration.
        
        Returns:
            Dict: Scheduler configuration
        """
        return self.config.get('scheduler', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get the logging configuration.
        
        Returns:
            Dict: Logging configuration
        """
        return self.config.get('logging', {})
    
    def get_optimizer_config(self) -> Dict[str, Any]:
        """
        Get the optimizer configuration.
        
        Returns:
            Dict: Optimizer configuration
        """
        return self.config.get('optimizer', {})
    
    def get_device(self) -> torch.device:
        """
        Get the device from the configuration or default to CUDA if available.
        
        Returns:
            torch.device: The device to use for training
        """
        device_name = self.get_training_config().get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device_name)
    
    def save_config(self, path: Optional[str] = None) -> None:
        """
        Save the configuration to a YAML file.
        
        Args:
            path: Path to save the configuration file, defaults to the original config path
        """
        save_path = path or self.config_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def __getitem__(self, key: str) -> Any:
        """
        Get an item from the configuration.
        
        Args:
            key: The key to get
            
        Returns:
            The value at the given key
        """
        if key in self.config:
            return self.config[key]
        
        # Handle flattened access (e.g., 'model.input_dim')
        if '.' in key:
            parts = key.split('.')
            curr = self.config
            for part in parts:
                if part in curr:
                    curr = curr[part]
                else:
                    raise KeyError(f"Key not found: {key}")
            return curr
            
        raise KeyError(f"Key not found: {key}")
    
    def update_from_dict(self, update_dict: Dict[str, Any]) -> None:
        """
        Update the configuration from a dictionary.
        
        Args:
            update_dict: Dictionary with new values to update
        """
        # Deep update function to handle nested dictionaries
        def deep_update(original, update):
            for key, value in update.items():
                if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                    deep_update(original[key], value)
                else:
                    original[key] = value
        
        deep_update(self.config, update_dict)
        
    def get_flat_config(self) -> Dict[str, Any]:
        """
        Get a flattened configuration dictionary.
        
        Returns:
            Dict: Flattened configuration dictionary
        """
        flat_config = {}
        
        def flatten(d, prefix=''):
            for k, v in d.items():
                if isinstance(v, dict):
                    flatten(v, prefix + k + '.')
                else:
                    flat_config[prefix + k] = v
        
        flatten(self.config)
        return flat_config 