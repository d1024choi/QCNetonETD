
import pytorch_lightning as pl
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from datasets import ETRIDataset
from transforms import TargetBuilder
import random

class ETRIDataModule(pl.LightningDataModule):

    def __init__(self,
                 root="./data",
                 train_batch_size=4,
                 val_batch_size=4,
                 test_batch_size=4,
                 shuffle=True,
                 num_workers=0,
                 pin_memory=True,
                 persistent_workers=True,
                 train_processed_dir=None,
                 val_processed_dir=None,
                 test_processed_dir=None,
                 train_transform=TargetBuilder(20, 60),
                 val_transform=TargetBuilder(20, 60),
                 test_transform=None,
                 **kwargs):
        super().__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_processed_dir = train_processed_dir
        self.val_processed_dir = val_processed_dir
        self.test_processed_dir = test_processed_dir
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def prepare_data(self):
        # No processing needed – just reading pickled files.
        pass

    def setup(self, stage=None):
        self.train_dataset = ETRIDataset(self.root, self.train_processed_dir, pre_transform=None, transform=self.train_transform)
        self.val_dataset = ETRIDataset(self.root, self.val_processed_dir, pre_transform=None, transform=self.val_transform)
        self.test_dataset = ETRIDataset( self.root, self.test_processed_dir, pre_transform=None, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)







