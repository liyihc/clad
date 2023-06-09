import csv
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, Field
from sklearn.metrics import roc_curve
from torch import nn, optim

from clad.utils import progress_bar

from ..common import OneHot, OnehotEmbedding, SeqDataLoader, get_onehot_infos
from .eval_dataloader import EvalDataLoader


class ModelInfo(BaseModel):
    onehots: List[OneHot] = []


class DeepAnTModel(nn.Module):
    def __init__(self, input_record_width, onehots: List[OneHot], bn=False) -> None:
        super().__init__()
        self.embedding = OnehotEmbedding(onehots)
        self.conv1 = nn.Conv1d(input_record_width +
                               self.embedding.output_dim, 30, 3, padding=1)
        self.conv2 = nn.Conv1d(30, 50, 3, padding=1)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(50, input_record_width + self.embedding.output_dim)

    def forward(self, x: torch.Tensor, onehots: torch.Tensor):
        """
        params:
            x include latest input
            x.shape == (batch, sequence, records width)
            onehots == (batch, sequence, onehots num)
        return:
            predict, actual
        """

        onehots = self.embedding(onehots)
        x = torch.cat((x, onehots), -1)
        actual = x[:, -1]
        x = x[:, :-1]
        x.swapdims_(1, 2)
        # x.shape == (batch, records width, sequence-1)
        x = torch.relu(torch.max_pool1d(self.conv1(x), 2, padding=1))
        x = torch.relu(self.avg_pool(self.conv2(x)))

        x = x.flatten(1, -1)
        predict = self.fc(x)
        return predict, actual


class SmallDeepAnTModel(nn.Module):
    def __init__(self, input_record_width, onehots: List[OneHot], bn: bool) -> None:
        super().__init__()
        self.embedding = OnehotEmbedding(onehots)
        dim = input_record_width + self.embedding.output_dim
        self.bn = nn.BatchNorm1d(dim) if bn else None
        self.conv1 = nn.Conv1d(dim, 20, 3, padding=1)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(20, input_record_width + self.embedding.output_dim)

    def forward(self, x: torch.Tensor, onehots: torch.Tensor):
        """
        params:
            x include latest input
            x.shape == (batch, sequence, records width)
            onehots == (batch, sequence, onehots num)
        return:
            predict, actual
        """

        onehots = self.embedding(onehots)
        x = torch.cat((x, onehots), -1)
        actual = x[:, -1]
        x = x[:, :-1]
        x.swapdims_(1, 2)

        if self.bn:
            x = self.bn(x)
        # x.shape == (batch, records width, sequence-1)
        x = torch.relu(self.avg_pool(self.conv1(x)))
        x = x.flatten(1, -1)
        predict = self.fc(x)
        return torch.relu(predict), actual


class Model(BaseModel):
    train_batch = 128
    sequence = 16
    epochs = 16
    lr = 1e-2
    lr_step_size = 4
    lr_step_gamma = .5
    device: str = "cuda:0"
    test_batch = 256
    user_limit = 0
    embedding = True
    batch_norm = False
    small_model = False

    def train(self, dataset_path: Path, output_path: Path):
        def loss_log(loss, right):
            with (output_path / "loss.csv").open('a') as f:
                writer = csv.writer(f)
                writer.writerow((loss, right))
        loss_log("loss", "right-rate")  # write header

        device = torch.device(self.device)

        record = pd.read_csv(dataset_path / "train-record.csv").to_numpy(float)
        onehots = pd.read_csv(dataset_path / "train-onehot.csv").to_numpy(int)
        values = [record, onehots]
        user = pd.read_csv(dataset_path / "train-user.csv").to_numpy(int).squeeze()
        onehot_infos = get_onehot_infos(onehots) if self.embedding else []
        info = ModelInfo(onehots=onehot_infos)
        with output_path.joinpath('model-info.json').open('w') as f:
            f.write(info.json(indent=4))

        input_size = record.shape[1]

        ModelType = DeepAnTModel if not self.small_model else SmallDeepAnTModel
        model = ModelType(input_size, info.onehots, self.batch_norm)
        model.to(device)

        criterion = nn.L1Loss()  # mae loss
        optimizer = optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=1e-5)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.5)

        model.train()
        dl = SeqDataLoader(values, user, self.sequence, self.train_batch)
        if self.user_limit:
            dl.pos = dl.pos[:min(self.user_limit, len(dl.pos))]
        with progress_bar(2) as (epochs_bar, train_bar):
            for _ in epochs_bar.range(self.epochs, description="epoch"):
                total_loss = 0
                total_right = 0
                cnt = 0

                xs: List[torch.Tensor]
                actual: torch.Tensor
                predict: torch.Tensor

                for xs, index, xs_ in train_bar.iter(dl, description="train"):
                    xs = [x.to(device) for x in xs]

                    optimizer.zero_grad()

                    predict, actual = model(*xs)
                    length = predict.shape[0]

                    loss: torch.Tensor = criterion(predict, actual)
                    loss.backward()
                    total_loss += loss.item() * length

                    optimizer.step()

                    total_right += length * loss
                    total_loss += loss.item() * length
                    cnt += length
                scheduler.step()
                loss_log(total_loss / cnt, total_right / cnt)
        model.cpu()
        torch.save(model.state_dict(), output_path / "model.pth")

    def test(self, dataset_path: Path, output_path: Path):
        device = torch.device(self.device)

        record = pd.read_csv(dataset_path / "test-record.csv").to_numpy(float)
        onehots = pd.read_csv(dataset_path / "test-onehot.csv").to_numpy(int)
        values = [record, onehots]
        user = pd.read_csv(dataset_path / "test-user.csv").to_numpy(int).squeeze()

        input_size = record.shape[1]

        info = ModelInfo.parse_file(output_path / 'model-info.json')
        ModelType = DeepAnTModel if not self.small_model else SmallDeepAnTModel
        model = ModelType(input_size, info.onehots, self.batch_norm)
        model.load_state_dict(torch.load(output_path / 'model.pth'))
        model.to(device)

        # dl = EvalDataLoader(values, user, self.sequence, self.test_batch)
        dl = SeqDataLoader(values, user, self.sequence, self.test_batch)
        if self.user_limit:
            dl.pos = dl.pos[:min(self.user_limit, len(dl.pos))]

        model.eval()
        mae = nn.L1Loss(reduction='sum')
        with torch.no_grad(), (output_path / "test-raw.csv").open('w') as f:
            writer = csv.writer(f)
            writer.writerow('user actual decision time'.split())
            actual: torch.Tensor
            predict: torch.Tensor
            with progress_bar() as (bar,):
                # for xs, xs_, index in bar.iter(dl):
                for xs, index , xs_ in bar.iter(dl):
                    xs = [x.to(device) for x in xs]
                    xs_ = [x.to(device) for x in xs_]

                    predict, actual = model(*xs)
                    writer.writerows((ind, 0, mae(pre, act).item())
                                     for ind, pre, act in zip(index, predict, actual))
                    xs[0][:, -1] = xs_[0]
                    xs[1][:, -1] = xs_[1]
                    predict, actual = model(*xs)
                    writer.writerows((ind, 1, mae(pre, act).item())
                                     for ind, pre, act in zip(index, predict, actual))

        raw = pd.read_csv(output_path / "test-raw.csv")

        fpr, tpr, threshold = roc_curve(raw["actual"], raw["decision"])
        threshold = threshold[(tpr - fpr).argmax()]

        raw['predict'] = (raw['decision'] > threshold).astype(int)
        raw.to_csv(output_path / 'test.csv', index=False)
