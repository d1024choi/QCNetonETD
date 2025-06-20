import os
import pickle
import torch
from typing import Optional, Callable
from torch_geometric.data import InMemoryDataset, HeteroData
from torch_geometric.transforms import BaseTransform


#New
"""
The code now directly loads the .pkl files and converts them into torch.pyg HeteroData objects, 
just like in the original QCNet implementation,
eliminating the need for any additional pre-processing
"""



from utils import wrap_angle

def dict_to_hetero_data(d):
    hetero = HeteroData()
    for key, value in d.items():
        # Handle edge types like ("map_point", "to", "map_polygon")
        if isinstance(key, tuple) and len(key) == 3:
            src, rel, dst = key
            hetero[(src, rel, dst)] = value
        # Handle node types like "agent", "map_point", etc.
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                hetero[key][subkey] = subvalue
        else:
            hetero[key] = value
    return hetero


class ETRIDataset(InMemoryDataset):
    def __init__(self,
                 root: str,
                 data_dir: str,
                 pre_transform: Optional[Callable] = None,
                 transform: Optional[Callable] = None):
        super().__init__(root=root, transform=transform, pre_transform=None, pre_filter=None)

        self._data_dir = os.path.join(root, data_dir)
        self.data, self.slices = self._load_data()

    def _load_data(self):

        if not os.path.exists(self._data_dir):
            raise FileNotFoundError(f"Data directory not found: {self._data_dir}")


        self.data_files = sorted([
            os.path.join(self._data_dir, f)
            for f in os.listdir(self._data_dir)
            if f.endswith('.pkl')
        ])

        data_list = []
        for file in self.data_files:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    data = dict_to_hetero_data(data)

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

        return self.collate(data_list)

    @property
    def processed_file_names(self):

        return []

    def _process(self):  
        pass


    def len(self):

        return len(self.slices[list(self.slices.keys())[0]]) - 1

    def get(self, idx):
        return super().get(idx)