from collections import defaultdict, deque
from math import ceil
from typing import List
import numpy as np
import torch


class EvalDataLoader:  # 只需要换一种数据产生方式即可了，保证每次都产生长度相等的即可，像seq_dataloader一样
    def __init__(self, xs: List[np.ndarray], y: np.ndarray, sequence: int = 16, batch: int = 32):
        self.xs = xs
        self.y = y
        self.sequence = sequence
        self.batch = batch

    def yield_single(self, indexes: List[List[int]]):
        indexes = indexes.copy()
        end_index = 2
        while True:
            start_index = max(0, end_index - self.sequence)
            for i in reversed(range(len(indexes))):
                l = indexes[i]
                if len(l) < end_index:
                    indexes.pop(i)
                    if not indexes:
                        return
                yield l[start_index:end_index]
            end_index += 1

    def __iter__(self):
        indexes = defaultdict(list)
        for ind, yy in enumerate(self.y):
            indexes[yy].append(ind)
        indexes = list(indexes.values())
        indexes.sort(key=len, reverse=True)

        batch = self.batch
        this_batch = []
        xs = self.xs
        y = self.y

        def handle_batch(this_batch):
            ind = np.array(this_batch)
            ret_x = [torch.from_numpy(x[ind]) for x in xs]
            ret_y = y[ind[:, 0]]

            pos = np.random.randint(0, len(y - 1), (len(ret_y),)) # 随机选择反例
            for i in range(3):
                same = ret_y == y[pos]
                l = same.sum()
                if not l:
                    break
                pos[same] = np.random.randint(0, len(y - 1), (l,))
            ret_x_ = [torch.from_numpy(x[pos]) for x in xs]
            return ret_x, ret_x_, ret_y

        for single in self.yield_single(indexes):
            if not this_batch or len(this_batch) < batch and len(this_batch[0]) == len(single):
                this_batch.append(single)
            else:
                yield handle_batch(this_batch)
                this_batch = [single]
        if this_batch:
            yield handle_batch(this_batch)

    def __len__(self):
        return ceil(len(self.y) / self.batch)
