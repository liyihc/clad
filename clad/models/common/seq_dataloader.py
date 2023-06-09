from math import ceil
import numpy as np
import torch

from typing import Dict, List, Tuple


def index_by(y: np.ndarray):
    index, pos_l = np.unique(y, return_index=True)
    pos_r = np.zeros_like(pos_l)
    pos_r[:-1] = pos_l[1:]
    pos_r[-1] = len(y)
    index = dict(zip(index, zip(pos_l, pos_r)))

    return index


class SeqDataLoader:
    '''
    输入时，x.shape = (n, feature), y.shape = (n,)
    在迭代(iter)时，返回x, y, x_，即记录、记录的标签及与标签不对应的其他记录
    x.shape = (batch, sequence_length, feature)
    '''

    def __init__(self, xs: List[np.ndarray], y: np.ndarray, sequence: int, batch: int = 32):
        assert all([len(x) == len(y) for x in xs])

        self.sequence = sequence
        self.batch = batch

        index = index_by(y)

        self.xs = xs
        self.y = y
        self.index: Dict[np.ndarray, Tuple[int, int]] = index
        pos = []

        for pos_l, pos_r in self.index.values():
            if pos_r - pos_l >= sequence:
                pos.append(np.arange(pos_l, pos_r - sequence + 1))

        self.pos = np.concatenate(pos)

    def __iter__(self):
        pos = np.random.permutation(self.pos)
        pos_ = np.random.randint(0, len(self.y) - 1, len(pos))
        for i in range(3):
            same = pos_ == pos
            pos_[same] = np.random.randint(0, len(self.y), same.sum())

        for index in range(0, len(pos), self.batch):
            end_index = min(index + self.batch, len(pos))
            indexes = pos[index:end_index].reshape(
                -1, 1) + np.arange(self.sequence)

            xs = [torch.from_numpy(x[indexes]) for x in self.xs]
            y = self.y[pos[index:end_index]]
            indexes_ = pos_[index:end_index]
            xs_ = [torch.from_numpy(x[indexes_]) for x in self.xs]
            yield (xs, y, xs_)

    def __len__(self):
        return ceil(len(self.pos) / self.batch)

    def len_element(self):
        return len(self.y)
