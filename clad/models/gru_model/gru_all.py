import csv
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, Field
from torch import nn, optim

from clad.utils import progress_bar

from ..common import OneHot, SeqDataLoader, get_onehot_infos
from .eval_dataloader import EvalDataLoader
from .profile_db import ProfileDB
from .torch_gru import TorchGru


class ModelInfo(BaseModel):
    onehots: List[OneHot] = []


class Model(BaseModel):
    train_batch = 128
    profile_size = 20
    sequence = 16
    epochs = 16
    lr = 1e-2
    lr_step_size = 4
    lr_step_gamma = .5
    device: str = "cuda:1"
    test_batch = 64
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

        record = pd.read_csv(dataset_path / "train-record.csv").to_numpy(float)
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

        input_size = record.shape[1]
        tf32 = torch.float32

        gru = TorchGru(self.profile_size, input_size, 2, info.onehots)
        gru.to(device)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(gru.parameters(), lr=self.lr, weight_decay=1e-5)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.5)

        profiledb = ProfileDB()
        default_profile = torch.zeros(self.profile_size, dtype=tf32)

        gru.train()
        dl = SeqDataLoader(values, user, self.sequence, self.train_batch)
        with progress_bar(2) as (epochs_bar, train_bar):
            for epoch in epochs_bar.range(self.epochs, description="epoch"):
                profiledb.clear_data()
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

                    profiles = profiledb.getProfiles(
                        index, default=default_profile).to(device)

                    optimizer.zero_grad()

                    y, ret_profiles = gru(xs[0], profiles, xs[1])
                    loss: torch.Tensor = criterion(y, torch.zeros(
                        y.shape[0], dtype=torch.int64, device=device))
                    loss.backward()
                    total_loss += loss.item() * y.shape[0]

                    xs[0][:, -1] = xs_[0]
                    xs[1][:, -1] = xs_[1]
                    y_, _ = gru(xs[0], profiles, xs[1])
                    loss = criterion(y_, torch.ones(
                        y.shape[0], dtype=torch.int64, device=device))
                    loss.backward()

                    optimizer.step()

                    total_right += y.shape[0] - y.argmax(1).sum().item()
                    total_wrong += y_.argmax(1).sum().item()
                    total_loss += loss.item() * y.shape[0]

                    cnt += y.shape[0]

                    profiledb.setProfiles(index, ret_profiles)

                scheduler.step()

                loss_log(total_loss / cnt, total_right /
                         cnt, total_wrong / cnt)

        gru.cpu()
        torch.save(gru.state_dict(), output_path.joinpath("gru.pth"))

    def test(self, dataset_path: Path, output_path: Path):
        device = torch.device(self.device)

        record = pd.read_csv(dataset_path / "test-record.csv").to_numpy(float)
        onehots = pd.read_csv(dataset_path / "test-onehot.csv").to_numpy(int)
        values = [record, onehots]
        user = pd.read_csv(dataset_path / "test-user.csv").to_numpy(int).squeeze()

        input_size = record.shape[1]

        tf32 = torch.float32

        profiledb = ProfileDB()
        default_profile = torch.zeros(self.profile_size, dtype=tf32)

        info = ModelInfo.parse_file(output_path.joinpath('model-info.json'))
        gru = TorchGru(self.profile_size, input_size, 2, info.onehots)
        print("load model")
        state_dict = torch.load(output_path.joinpath("gru.pth"))
        gru.load_state_dict(state_dict)
        gru.to(device)

        dl = EvalDataLoader(values, user, batch=self.test_batch)

        gru.eval()
        with torch.no_grad(), output_path.joinpath("test.csv").open('w') as f:
            writer = csv.writer(f)
            writer.writerow(('user', 'actual', 'decision', 'predict'))
            y: torch.Tensor
            y_: torch.Tensor
            with progress_bar() as (bar,):
                for xs, index in bar.iter(dl):
                    xs = [x.to(device) for x in xs]

                    profiles = profiledb.getProfiles(
                        index, default=default_profile).to(device)

                    y, ret_profiles = gru(xs[0], profiles, xs[1])
                    profiledb.setProfiles(index, ret_profiles)

                    profiles = profiledb.getWrongProfiles(
                        index, default=default_profile).to(device)
                    y_, _ = gru(xs[0], profiles, xs[1])

                    writer.writerows((ind, 0, (yy[1] - yy[0]).item(), yy.argmax().item())
                                     for ind, yy in zip(index, y.cpu()))
                    writer.writerows((ind, 1, (yy[1] - yy[0]).item(), yy.argmax().item())
                                     for ind, yy in zip(index, y_.cpu()))
