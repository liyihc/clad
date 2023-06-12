from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd


class DataReader:
    def __init__(self, dataset_path: Path, mode: Literal["train", "test"]):
        self.record: np.ndarray = pd.read_csv(
            dataset_path / f"{mode}-record.csv").to_numpy(np.float32)
        self.onehot: np.ndarray = pd.read_csv(
            dataset_path / f"{mode}-onehot.csv").to_numpy(int)
        self.user: np.ndarray = pd.read_csv(
            dataset_path / f"{mode}-user.csv").to_numpy(int).squeeze()
        
    @property
    def values(self):
        return [self.record, self.onehot]
