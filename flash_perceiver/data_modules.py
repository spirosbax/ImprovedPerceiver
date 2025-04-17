import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms

class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, config_manager):
        super().__init__()
        
        # Store configuration
        self.config_manager = config_manager
        
        # Extract configurations
        self.training_config = config_manager.get_training_config()
        self.augmentation_config = config_manager.get_augmentation_config()
        
        # Set default paths and parameters
        self.data_root = self.training_config.get('data_root', '../data')
        self.batch_size = self.training_config.get('batch_size', 512)
        self.num_workers = self.training_config.get('num_workers', 4)
        
        # Set transforms
        self.transform_train = None
        self.transform_test = None
        
        # Dataset properties
        self.train_set = None
        self.val_set = None
        self.test_set = None
        
    def setup(self, stage=None):
        # Define transforms
        self._setup_transforms()
        
        # Load datasets if not already loaded
        if self.train_set is None and self.val_set is None and self.test_set is None:
            # Load full training set
            full_train_set = CIFAR100(self.data_root, train=True, download=True, transform=self.transform_train)
            
            # Split into train and validation sets
            val_size = int(len(full_train_set) * 0.1)  # 10% for validation
            train_size = len(full_train_set) - val_size
            self.train_set, self.val_set = random_split(full_train_set, [train_size, val_size])
            
            # Load test set
            self.test_set = CIFAR100(self.data_root, train=False, download=True, transform=self.transform_test)
            
            # Print dataset information
            print("\n==> Dataset Information:")
            print(f"    - Training samples: {len(self.train_set)}")
            print(f"    - Validation samples: {len(self.val_set)}")
            print(f"    - Test samples: {len(self.test_set)}")
            print(f"    - Classes: 100")
            print("")
    
    def _setup_transforms(self):
        """Setup data transformations based on the configuration."""
        transform_train = []
        
        augmentation_type = self.augmentation_config.get('type', 'standard')  # Default to standard augmentation
        
        if augmentation_type.lower() == 'moco':
            # MoCo-style augmentation
            print("==> Using MoCo-style augmentation for training data:")
            print(f"    - RandomResizedCrop with scale {self.augmentation_config.get('resize_scale', (0.2, 1.0))}")
            print(f"    - RandomGrayscale with p={self.augmentation_config.get('grayscale_p', 0.2)}")
            print(f"    - ColorJitter with brightness={self.augmentation_config.get('jitter_brightness', 0.4)}, "
                  f"contrast={self.augmentation_config.get('jitter_contrast', 0.4)}, "
                  f"saturation={self.augmentation_config.get('jitter_saturation', 0.4)}, "
                  f"hue={self.augmentation_config.get('jitter_hue', 0.4)}")
            print(f"    - RandomHorizontalFlip")
            print("")
            
            # Create MoCo transforms - Note: We're adapting to CIFAR scale
            transform_train.extend([
                transforms.RandomResizedCrop(32, scale=self.augmentation_config.get('resize_scale', (0.2, 1.0))),
                transforms.RandomGrayscale(p=self.augmentation_config.get('grayscale_p', 0.2)),
                transforms.ColorJitter(
                    brightness=self.augmentation_config.get('jitter_brightness', 0.4),
                    contrast=self.augmentation_config.get('jitter_contrast', 0.4),
                    saturation=self.augmentation_config.get('jitter_saturation', 0.4),
                    hue=self.augmentation_config.get('jitter_hue', 0.4)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif augmentation_type.lower() == 'randaug' or self.augmentation_config.get('use_randaug', False):
            # RandAugment-based augmentation
            randaug_n = self.augmentation_config.get('randaug_n', 2)
            randaug_m = self.augmentation_config.get('randaug_m', 9)
            print("==> Using RandAugment for training data:")
            print(f"    - N = {randaug_n} (number of augmentations applied sequentially)")
            print(f"    - M = {randaug_m} (magnitude of augmentations, range 0-30)")
            print("")
            
            # Add RandAugment transform
            transform_train.append(transforms.RandAugment(num_ops=randaug_n, magnitude=randaug_m))
            
            # Add standard augmentations
            transform_train.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            # Standard augmentation
            print("==> Using standard data augmentation (RandomCrop + HorizontalFlip)")
            print("")
            transform_train.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        
        self.transform_train = transforms.Compose(transform_train)
        
        # Test transforms - always the same
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    def train_dataloader(self):
        return DataLoader(
            self.train_set, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_set, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_set, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=True
        ) 