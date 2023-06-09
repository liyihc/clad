import csv
from pathlib import Path
from typing import List

import pandas as pd
import torch
from pydantic import BaseModel, Field
from sklearn.metrics import roc_curve
from torch import nn, optim

from clad.utils import progress_bar

from ..cnn_model.eval_dataloader import EvalDataLoader as CNNEvalDataLoader
from ..common import OneHot, OnehotEmbedding, SeqDataLoader, get_onehot_infos


class ModelInfo(BaseModel):
    onehots: List[OneHot] = []


class GruAeModel(nn.Module):
    """

    """

    def __init__(self, input_record_width, onehots: List[OneHot]) -> None:
        super().__init__()
        self.embeding = OnehotEmbedding(onehots)
        self.gru1 = nn.GRU(
            input_size=input_record_width + self.embeding.output_dim,
            hidden_size=200,
            batch_first=True)
        self.gru2 = nn.GRU(
            input_size=200,
            hidden_size=200,
            batch_first=True
        )
        self.output_fc = nn.Linear(
            200, input_record_width + self.embeding.output_dim)

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
        # (batch, dims)
        x = torch.stack([x] * length, 1)
        x = torch.tanh(self.gru2(x)[0])
        # (batch, sequence, 200)
        return origin, torch.tanh(self.output_fc(x))


class Model(BaseModel):
    train_batch = 128
    sequence = 16
    epochs = 16
    lr = 1e-2
    lr_step_size = 4
    lr_step_gamma = .5
    device: str = Field("cuda:0", compare=False)
    test_batch = Field(256, compare=False)
    skip_small_seq = False

    def train(self, dataset_path: Path, output_path: Path):
        def loss_log(loss):
            with output_path.joinpath("loss.csv").open('a') as f:
                writer = csv.writer(f)
                writer.writerow((loss, ))

        with output_path.joinpath("loss.csv").open('w') as f:
            writer = csv.writer(f)
            writer.writerow(("loss",))

        device = torch.device(self.device)

        record = pd.read_csv(
            dataset_path / "train-record.csv").to_numpy(float)
        onehots = pd.read_csv(dataset_path / "train-onehot.csv").to_numpy(int)
        values = [record, onehots]
        user = pd.read_csv(dataset_path / "train-user.csv").to_numpy(int).squeeze()

        info = ModelInfo(onehots=get_onehot_infos(onehots))
        with output_path.joinpath('model-info.json').open('w') as f:
            f.write(info.json())

        model = GruAeModel(record.shape[1], info.onehots)
        model.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=1e-5)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.5)

        model.train()
        dl = SeqDataLoader(values, user, self.sequence, self.train_batch)
        with progress_bar(2) as (epochs_bar, train_bar):
            for epoch in epochs_bar.range(self.epochs, description="epoch"):
                total_loss = 0
                cnt = 0

                xs: List[torch.Tensor]
                loss: torch.Tensor
                origin: torch.Tensor
                decode: torch.Tensor
                for xs, index, _ in train_bar.iter(dl, "train"):
                    xs = [x.to(device) for x in xs]

                    optimizer.zero_grad()

                    origin, decode = model(*xs)
                    loss = criterion(origin, decode)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * origin.shape[0]
                    cnt += origin.shape[0]
                scheduler.step()
                loss_log(total_loss / cnt)

        model.cpu()
        torch.save(model.state_dict(), output_path.joinpath("model.pth"))

    def test(self, dataset_path: Path, output_path: Path):
        device = torch.device(self.device)

        record = pd.read_csv(dataset_path / "test-record.csv").to_numpy(float)
        onehots = pd.read_csv(dataset_path / "test-onehot.csv").to_numpy(int)
        values = [record, onehots]
        user = pd.read_csv(dataset_path / "test-user.csv").to_numpy(int).squeeze()


        info = ModelInfo.parse_file(output_path.joinpath('model-info.json'))
        model = GruAeModel(record.shape[1], info.onehots)
        model.load_state_dict(torch.load(output_path.joinpath("model.pth")))
        model.to(device)

        dl = CNNEvalDataLoader(values, user, self.sequence, batch=self.test_batch)

        model.eval()
        with torch.no_grad(), output_path.joinpath("test raw.csv").open('w') as f:
            writer = csv.writer(f)
            writer.writerow(('user', 'actual', 'decision'))
            with progress_bar() as (bar,):
                origin: torch.Tensor
                decode: torch.Tensor
                for xs, xs_, index in bar.iter(dl):
                    if self.skip_small_seq and len(xs[0]) < self.sequence:
                        continue
                    xs = [x.to(device) for x in xs]
                    xs_ = [x.to(device) for x in xs_]

                    origin, decode = model(*xs)
                    bias = ((decode[:, -1] - origin[:, -1])**2).sum(1)
                    xs[0][:, -1] = xs_[0]
                    xs[1][:, -1] = xs_[1]
                    origin, decode = model(*xs)
                    bias_ = ((decode[:, -1] - origin[:, -1])**2).sum(1)

                    writer.writerows((ind, 0, b.item())
                                     for ind, b in zip(index, bias.cpu()))
                    writer.writerows((ind, 1, b.item())
                                     for ind, b in zip(index, bias_.cpu()))

        data = pd.read_csv(output_path.joinpath("test raw.csv"))
        fpr, tpr, threshold = roc_curve(data['actual'], data['decision'])

        argmax = (tpr - fpr).argmax()
        threshold = threshold[argmax]
        data['predict'] = (data['decision'] > threshold).astype(int)
        data.to_csv(output_path.joinpath("test.csv"), index=False)
