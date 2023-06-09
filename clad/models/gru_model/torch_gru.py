from typing import List
from pydantic import BaseModel
import torch
from torch import nn
from ..common import OneHot, OnehotEmbedding



class TorchGru(nn.Module):
    def __init__(self, hidden_width, input_width, output_width, onehots: List[OneHot]):
        super().__init__()

        self.embedding = OnehotEmbedding(onehots)

        self.hidden_width = hidden_width
        self.input_width = input_width
        self.output_width = output_width

        actual_input = input_width + self.embedding.output_dim
        self.gru = nn.GRU(actual_input, hidden_width, batch_first=True)
        self.fc = nn.Linear(hidden_width, output_width)

    def forward(self, x, h_0, onehots):
        """
        params
        @x: [batchsize, sequence_size, inputwidth]
        @h_0: [batchsize, hiddenwidth]
        @onehots: [batchsize, sequence_size, onehot_number]

        return
            output: [batchsize, outputwidth]
            h_t: [batchsize, hiddenwidth]
        """
        x = torch.cat([x, self.embedding(onehots)], 2)

        h_0 = h_0.view(1, *h_0.shape)
        output, h_t = self.gru(x, h_0)
        output = self.fc(torch.tanh(output[:, -1]))
        return torch.log_softmax(output, dim=1), h_t[0]

    def single_forward(self, x, h):
        x = x.view(x.shape[0], 1, *x.shape[1:])
        return self.forward(x, h)
