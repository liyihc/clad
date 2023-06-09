from collections import defaultdict, deque
from math import ceil
from typing import List
import numpy as np
import torch


class EvalDataLoader:
    def __init__(self, xs: List[np.ndarray], y: np.ndarray, batch: int = 32):
        self.xs = xs
        self.y = y
        self.batch = batch

    def __iter__(self):
        indexes = defaultdict(deque)
        for ind, yy in enumerate(self.y):
            indexes[yy].append(ind)
        indexes = list(indexes.values())
        indexes.sort(key=len, reverse=True)

        batch = self.batch
        while indexes:
            this_batch = []
            for i in reversed(range(batch)):
                q = indexes[i]
                this_batch.append(q.popleft())
                if not q:
                    indexes.pop(i)
                    if batch > len(indexes):
                        batch = len(indexes)

            this_batch = np.array(this_batch)
            ret_x = [torch.unsqueeze(torch.from_numpy(
                x[this_batch]), 1) for x in self.xs]
            ret_y = self.y[this_batch]
            yield ret_x, ret_y

    def __len__(self):
        return ceil(len(self.xs[0]) / self.batch)
