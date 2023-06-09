import csv
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from pydantic import BaseModel

from clad.utils import progress_bar
from ..common import OneHot, OnehotEmbedding, get_onehot_infos, SeqDataLoader
from .eval_dataloader import EvalDataLoader


class ModelInfo(BaseModel):
    onehots: List[OneHot] = []


class CNNAllModel(nn.Module):
    def __init__(self, input_record_width, onehots: List[OneHot]) -> None:
        super().__init__()
        self.embeding = OnehotEmbedding(onehots)
        self.conv1 = nn.Conv1d(input_record_width +
                               self.embeding.output_dim, 30, 3, padding=1)
        self.conv2 = nn.Conv1d(30, 50, 3, padding=1)
        self.conv3 = nn.Conv1d(50, 50, 3, padding=1)

        self.avg_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(50, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x: torch.Tensor, onehots: torch.Tensor):
        """
        x.shape == (batch, sequence, records)
        onehots == (batch, sequence, onehots num)

        output: (batch, 2)
        """
        onehots = self.embeding(onehots)
        x = torch.cat((x, onehots), -1)
        x.swapdims_(1, 2)
        # (batch, records, sequence)
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        x = self.avg_pool(x).flatten(1, -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return torch.log_softmax(x, dim=1)


class Model(BaseModel):
    train_batch = 128
    sequence = 16
    epochs = 16
    lr = 1e-2
    lr_step_size = 4
    lr_step_gamma = .5
    device: str = "cuda:0"
    test_batch = 256
    embedding = True

    def train(self, dataset_path: Path, output_path: Path):
        def loss_log(loss, right, wrong):
            with output_path.joinpath("loss.csv").open('a') as f:
                writer = csv.writer(f)
                writer.writerow((loss, right, wrong))

        with output_path.joinpath("loss.csv").open('w') as f:
            writer = csv.writer(f)
            writer.writerow(("loss", "right-right", "wrong-right"))

        device = torch.device(self.device)

        record = pd.read_csv(dataset_path / "train-record.csv").to_numpy(np.float32)
        onehots = pd.read_csv(dataset_path / "train-onehot.csv").to_numpy(int)
        values = [record, onehots]
        user = pd.read_csv(dataset_path / "train-user.csv").to_numpy(int).squeeze()

        if self.embedding:
            onehot_infos = get_onehot_infos(onehots)
        else:
            onehot_infos = []
        info = ModelInfo(onehots=onehot_infos)
        with output_path.joinpath('model-info.json').open('w') as f:
            f.write(info.json())

        model = CNNAllModel(record.shape[1], info.onehots)
        model.to(device)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=1e-5)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.5)

        model.train()
        dl = SeqDataLoader(values, user, self.sequence, self.train_batch)
        with progress_bar(2) as (epochs_bar, train_bar):
            for epoch in epochs_bar.range(self.epochs, description="epoch"):
                total_loss = 0
                total_right = 0
                total_wrong = 0
                cnt = 0

                xs: List[torch.Tensor]
                xs_: List[torch.Tensor]
                y: torch.Tensor
                y_: torch.Tensor
                for xs, index, xs_ in train_bar.iter(dl):
                    xs = [x.to(device) for x in xs]
                    xs_ = [x.to(device) for x in xs_]

                    optimizer.zero_grad()

                    y = model(*xs)
                    loss: torch.Tensor = criterion(y, torch.zeros(
                        y.shape[0], dtype=torch.int64, device=device))
                    loss.backward()
                    total_loss += loss.item() * y.shape[0]

                    xs[0][:, -1] = xs_[0]
                    xs[1][:, -1] = xs_[1]

                    y_ = model(*xs)
                    loss = criterion(y_, torch.ones(
                        y.shape[0], dtype=torch.int64, device=device))
                    loss.backward()

                    optimizer.step()

                    total_right += y.shape[0] - y.argmax(1).sum().item()
                    total_wrong += y_.argmax(1).sum().item()
                    total_loss += loss.item() * y.shape[0]

                    cnt += y.shape[0]

                scheduler.step()

                loss_log(total_loss / cnt, total_right /
                         cnt, total_wrong / cnt)

        model.cpu()
        torch.save(model.state_dict(), output_path.joinpath("model.pth"))

    def test(self, dataset_path: Path, output_path: Path):
        device = torch.device(self.device)

        record = pd.read_csv(dataset_path / "train-record.csv").to_numpy(float)
        onehots = pd.read_csv(dataset_path / "train-onehot.csv").to_numpy(int)
        values = [record, onehots]
        user = pd.read_csv(dataset_path / "train-user.csv").to_numpy(int).squeeze()

        info = ModelInfo.parse_file(output_path.joinpath('model-info.json'))
        model = CNNAllModel(record.shape[1], info.onehots)
        model.load_state_dict(torch.load(output_path.joinpath("model.pth")))
        model.to(device)

        dl = EvalDataLoader(values, user, self.sequence, batch=self.test_batch)

        model.eval()
        with torch.no_grad(), output_path.joinpath("test.csv").open('w') as f:
            writer = csv.writer(f)
            writer.writerow(('user', 'actual', 'decision', 'predict'))
            y: torch.Tensor
            y_: torch.Tensor
            with progress_bar() as (bar,):
                for xs, xs_, index in bar.iter(dl):
                    xs = [x.to(device) for x in xs]
                    xs_ = [x.to(device) for x in xs_]

                    y = model(xs[0], xs[1])
                    xs[0][:, -1] = xs_[0]
                    xs[1][:, -1] = xs_[1]
                    y_ = model(xs[0], xs[1])

                    writer.writerows((ind, 0, (yy[1] - yy[0]).item(), yy.argmax().item())
                                     for ind, yy in zip(index, y.cpu()))
                    writer.writerows((ind, 1, (yy[1] - yy[0]).item(), yy.argmax().item())
                                     for ind, yy in zip(index, y_.cpu()))
