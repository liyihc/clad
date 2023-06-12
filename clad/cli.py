from datetime import datetime
from enum import Enum
import json
from pathlib import Path
from traceback import format_exc
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score
import typer

app = typer.Typer(no_args_is_help=True)

from .models import models, DataReader


class Dataset(str, Enum):
    eshop = "eshop"
    yoochoose = "yoochoose"


class Model(str, Enum):
    cnn_all = "cnn-all"
    cnn_gm = "cnn-gm"
    cnn_cll = "cnn-cll"
    cnn = "cnn"
    gru_all = "gru-all"
    gru_gm = "gru-gm"
    gru_cll = "gru-cll"
    gru = "gru"


class Params(BaseModel):
    dataset: Dataset
    model: Model
    model_params: dict = {}

    def get_model(self):
        return models[self.model.value].parse_obj(self.model_params)


@app.command("get-config", no_args_is_help=True)
def get_config(
        dataset: Dataset,
        model: Model):
    m = models[model]
    params = Params(
        dataset=dataset,
        model=model)
    params.model_params = m().dict()
    print(params.json(indent=4))


@app.command("run", no_args_is_help=True)
def run(
    config_file: Path = typer.Argument(...,
                                       exists=True, dir_okay=False, readable=True, resolve_path=True),
    output_parent_folder: Path = typer.Option(
        Path("output"),
        "-o", "--output"),
    data_folder: Path = typer.Option(
        Path("data"),
        "-d", "--data", help="dataset to be fed into model"),
):
    params = Params.parse_file(config_file)
    model = params.get_model()
    params.model_params = model.dict()
    output_folder = output_parent_folder / \
        f"{params.dataset} {params.model} {datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_folder.mkdir(parents=True, exist_ok=True)
    dataset_folder = data_folder / params.dataset.value

    try:
        start = datetime.now()
        (output_folder / "params.json").write_text(params.json(indent=4))
        print("training")
        model.train(DataReader(dataset_folder, "train"), output_folder)
        print("testing")
        model.test(DataReader(dataset_folder, "test"), output_folder)

        print("calc auc score")
        csv = pd.read_csv(output_folder / "test.csv")
        user_auc = [roc_auc_score(d["actual"], d["decision"]) for user, d in csv.groupby('user')]
        avg_auc = np.mean(user_auc)
        td = datetime.now() - start

        (output_folder / "result.json").write_text(json.dumps({
            "avg_auc": avg_auc,
            "minutes": td.total_seconds() / 60
        }, indent=4))
        print("average auc score:", avg_auc)
        print("running time (minutes):", td.total_seconds() / 60)

    except Exception:
        (output_folder / "error.txt").write_text(format_exc())
        from rich.console import Console
        console = Console()
        console.print_exception()
