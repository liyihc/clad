import csv
from pathlib import Path
from typing import Callable, List, Literal

import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from pydantic import BaseModel, Field

from clad.utils import progress_bar, Bar
from ..common import OneHot, OnehotEmbedding, get_onehot_infos, DataToSequences, UserDataLoader


class GruAeModel(nn.Module):
    """

    """

    def __init__(self, input_record_width, onehots: List[OneHot]) -> None:
        super().__init__()
        self.embeding = OnehotEmbedding(onehots)
        input_size = input_record_width + self.embeding.output_dim
        self.gru1 = nn.GRU(
            input_size=input_size,
            hidden_size=20,
            batch_first=True)
        self.gru2 = nn.GRU(
            input_size=input_size,
            hidden_size=20,
            batch_first=True
        )
        self.output_fc = nn.Linear(
            20, input_size)

    def forward(self, x: torch.Tensor, onehots: torch.Tensor):
        """
        x.shape == (batch, sequence, records)
        onehots == (batch, sequence, onehots num)

        output: (batch, sequence, feature num), (batch, sequence, feature num)
        """
        onehots = self.embeding(onehots)
        length = x.shape[1]
        x = torch.cat((x, onehots), -1)
        origin = x

        xs = []
        h = torch.tanh(self.gru1(x)[1])  # (1, batch, hidden)
        x = torch.tanh(self.output_fc(h.squeeze(0)))  # (batch, record)
        xs.append(x)
        for _ in range(length - 1):
            h = torch.tanh(self.gru2(x.unsqueeze(1), h)[1])
            x = torch.tanh(self.output_fc(h.squeeze(0)))
            xs.append(x)

        xs = torch.stack(xs, 1)  # (batch, sequence, records)
        return origin, xs.flip(1)


class GruCLLModel(nn.Module):
    def __init__(self, input_record_width, onehots: List[OneHot]) -> None:
        super().__init__()
        self.embeding = OnehotEmbedding(onehots)
        self.gru1 = nn.GRU(
            input_size=input_record_width + self.embeding.output_dim,
            hidden_size=20,
            batch_first=True)
        self.output_fc = nn.Linear(
            20, 2)

    def forward(self, x: torch.Tensor, onehots: torch.Tensor):
        """
        x.shape == (batch, sequence, records)
        onehots == (batch, sequence, onehots num)

        output: (batch, sequence, feature num), (batch, sequence, feature num)
        """
        onehots = self.embeding(onehots)
        length = x.shape[1]
        x = torch.cat((x, onehots), -1)
        origin = x

        x = torch.tanh(self.gru1(x)[1][0])
        # (batch, sequence, width)
        return torch.log_softmax(self.output_fc(x), dim=1)


class Model(BaseModel):
    lr = .1
    sequence = 5

    max_records = 10000
    epochs = 64
    train_batch = 64
    test_batch = 1024
    user_limit = -1
    fix_anomaly = True
    embedding = True

    device: str = "cuda:0"

    def get_model(self, *args, **kwds):
        return GruCLLModel(*args, **kwds)

    def get_epochs(self, record_num: int):
        return max(1, min(self.epochs, self.max_records // record_num))

    def test(self, dataset_path: Path, output_path: Path):
        record = pd.read_csv(dataset_path / "test-record.csv").to_numpy(float)
        onehot = pd.read_csv(dataset_path / "test-onehot.csv").to_numpy(int)
        user = pd.read_csv(dataset_path / "test-user.csv").to_numpy(int).squeeze()
        onehot_infos = get_onehot_infos(onehot) if self.embedding else []

        dl = UserDataLoader([record, onehot], user)

        with output_path.joinpath("test.csv").open('w') as f:
            writer = csv.writer(f)
            writer.writerow(('user', 'actual', 'decision'))

        input_size = record.shape[1]

        with progress_bar(2) as (user_bar, epochs_bar):
            for i, (x, y, anomaly_generator) in enumerate(user_bar.iter(dl)):
                if i > self.user_limit > 0:
                    break
                length = len(x[0])

                model = self.get_model(input_size, onehot_infos)

                train_data = [xx[length // 3:] for xx in x]
                self._train(
                    model, train_data, anomaly_generator, epochs_bar)
                test_data = [xx[:length // 3] for xx in x]
                test_data_ = anomaly_generator(len(test_data[0]))
                result = self._test(
                    model, test_data, test_data_)
                with output_path.joinpath("test.csv").open('a') as f:
                    writer = csv.writer(f)
                    writer.writerows((y, *d) for d in result)

    def _train(self, model: GruCLLModel, xs: List[np.ndarray], anomaly_generator: Callable[[int], List[np.ndarray]], epoch_bar: Bar):
        device = torch.device(self.device)

        model.to(device)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=1e-5)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.5)
        model.train()

        losses = []
        if self.fix_anomaly:
            xs_ = anomaly_generator(len(xs[0]))
            dl = DataToSequences([*xs, *xs_], [1, 1, 0, 0],
                                 self.sequence, self.train_batch)
        for epoch in epoch_bar.range(self.get_epochs(len(xs[0])), description="epoch"):
            total_loss = 0
            cnt = 0

            loss: torch.Tensor
            if not self.fix_anomaly:
                xs_ = anomaly_generator(len(xs[0]))
                dl = DataToSequences([*xs, *xs_], [1, 1, 0, 0],
                                     self.sequence, self.train_batch)
            for data in dl:
                r, o, r_, o_ = (d.to(device) for d in data)

                optimizer.zero_grad()

                # print(r.shape, o.shape)
                y = model(r, o)
                loss = criterion(y, torch.zeros(
                    y.shape[0], dtype=torch.int64, device=device))
                loss.backward()
                total_loss += loss.item() * y.shape[0]

                r[:, -1] = r_
                o[:, -1] = o_

                y_ = model(r, o)
                loss = criterion(y_, torch.ones(
                    y.shape[0], dtype=torch.int64, device=device))
                loss.backward()

                optimizer.step()

                total_loss += loss.item() * y.shape[0]

                cnt += y.shape[0]

            if losses and losses[-1] * 0.9 <= total_loss / cnt:
                scheduler.step()

    def _test(self, model: GruCLLModel, xs: List[np.ndarray], xs_: List[np.ndarray]):
        device = torch.device(self.device)
        model.to(device)

        dl = DataToSequences([*xs, *xs_], [1, 1, 0, 0],
                             self.sequence, self.test_batch)
        model.eval()
        with torch.no_grad():
            results = []
            for data in dl:
                r, o, r_, o_ = (d.to(device) for d in data)
                predict = model(r, o)
                # actual, decision
                results.extend(
                    (0, (p[1] - p[0]).item()) for p in predict.cpu())

                r[:, -1] = r_
                o[:, -1] = o_
                predict = model(r, o)
                results.extend(
                    (1, (p[1] - p[0]).item()) for p in predict.cpu())
            return results

class ModelAE(Model):
    def get_model(self, *args, **kwds):
        return GruAeModel(*args, **kwds)

    def _train(self, model: GruCLLModel, xs: List[np.ndarray], anomaly_generator: Callable[[int], List[np.ndarray]], epoch_bar: Bar):
        device = torch.device(self.device)

        model.to(device)

        criterion = nn.L1Loss()
        optimizer = optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=1e-5)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.5)
        model.train()

        dl = DataToSequences(xs, [1, 1], self.sequence, self.train_batch)
        losses = []
        for epoch in epoch_bar.range(self.get_epochs(len(dl)), description="epoch"):
            total_loss = 0
            cnt = 0

            loss: torch.Tensor
            for data in dl:
                r, o = (d.to(device) for d in data)

                optimizer.zero_grad()

                predict, actual = model(r, o)

                loss = criterion(predict, actual)
                loss.backward()

                optimizer.step()

                total_loss += loss.item() * predict.shape[0]

                cnt += predict.shape[0]

            if losses and losses[-1] * 0.9 <= total_loss / cnt:
                scheduler.step()

    def _test(self, model: GruAeModel, xs: List[np.ndarray], xs_: List[np.ndarray]):
        device = torch.device(self.device)
        model.to(device)

        dl = DataToSequences([*xs, *xs_], [1, 1, 0, 0],
                             self.sequence, self.test_batch)
        model.eval()
        mae = nn.L1Loss(reduction='sum')
        with torch.no_grad():
            results = []
            for data in dl:
                r, o, r_, o_ = (d.to(device) for d in data)
                # print(r.shape, o.shape, r_.shape, o_.shape)
                predict, actual = model(r, o)
                # actual, decision
                results.extend(
                    (0, (mae(pre, act)).item()) for pre, act in zip(predict, actual))

                r[:, -1] = r_
                o[:, -1] = o_
                predict, actual = model(r, o)
                results.extend(
                    (1, (mae(pre, act)).item()) for pre, act in zip(predict, actual))
            return results
