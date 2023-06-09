from math import ceil
import numpy as np
import pandas as pd

from typing import Dict, List, Tuple, Union

import torch


def index_by(y: np.ndarray):
    index, pos_l = np.unique(y, return_index=True)
    pos_r = np.zeros_like(pos_l)
    pos_r[:-1] = pos_l[1:]
    pos_r[-1] = len(y)
    index = dict(zip(index, zip(pos_l, pos_r)))

    return index


class UserDataLoader:
    '''
    输入时，x.shape = (n, feature), y.shape = (n,)
    在迭代(iter)时，返回x, y, x_，即记录、记录的标签及与标签不对应的其他记录
    x.shape = (batch, sequence_length, feature)
    '''

    def __init__(self, xs: List[np.ndarray], y: np.ndarray):
        arg = y.argsort(kind='stable')  # first, group by user
        xs = [x[arg] for x in xs]
        y = y[arg]
        index = index_by(y)

        self.xs = xs
        self.y = y
        self.index: Dict[np.ndarray, Tuple[int, int]] = index

    def __iter__(self):
        for index, (l, r) in self.index.items():
            def generate_anomalies(num: int) -> List[np.ndarray]:
                x_ = np.random.randint(0, len(self.y) - (r - l), num)
                x_[x_ >= l] += r - l
                return [x[x_] for x in self.xs]
            yield [x[l:r] for x in self.xs], self.y[l], generate_anomalies

    def __len__(self):
        return len(self.index)

    def len_element(self):
        return len(self.xs[0])


class DataToSequences:
    def __init__(self, xs: List[np.ndarray], is_seq: List[Union[bool, int]], sequence: int, batch: int) -> None:
        self.xs = xs
        self.is_seq = is_seq
        self.sequence = sequence
        self.batch = batch

    def __iter__(self):
        pos = np.random.permutation(self.xs[0].shape[0] - self.sequence + 1)

        batch = self.batch
        for index in range(0, len(pos), batch):
            end_index = min(index + batch, len(pos))
            indexes = pos[index:end_index]
            seq_indexes = indexes.reshape(-1, 1) + np.arange(self.sequence)

            yield [torch.from_numpy(x[seq_indexes] if is_s else x[indexes]) for x, is_s in zip(self.xs, self.is_seq)]

    def __len__(self):
        return ceil((self.xs[0].shape[0] - self.sequence + 1) / self.batch)
