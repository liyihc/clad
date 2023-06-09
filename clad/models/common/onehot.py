from datetime import timedelta, timezone, datetime
from typing import List

import numpy as np
from pydantic import BaseModel
import math
import torch
from torch import nn

tz = timezone(timedelta(hours=8))


class OneHot(BaseModel):
    numbers: int
    dim: int


def get_onehot_infos(onehots: np.ndarray) -> List[OneHot]:
    onehot_infos = []
    for ind in range(onehots.shape[1]):
        onehot: np.ndarray = onehots[:, ind]
        num = onehot.max() + 1
        num += num // 5
        onehot_infos.append(
            OneHot(numbers=num, dim=int(math.log10(num) + 1)))

    return onehot_infos


class OnehotEmbedding(nn.Module):
    def __init__(self, onehots: List[OneHot]) -> None:
        super().__init__()
        self.m_list = nn.ModuleList(
            nn.Embedding(o.numbers, o.dim) for o in onehots)
        self.output_dim = sum(o.dim for o in onehots)

    def forward(self, onehots: torch.Tensor):
        """
        input: (*, onehot numbers)
        output: (*, onehot dim)
        """
        if self.output_dim:
            return torch.cat([m(onehots[..., i])for i, m in enumerate(self.m_list)], -1)
        return torch.empty((*onehots.shape[:-1], 0), device=onehots.device)
