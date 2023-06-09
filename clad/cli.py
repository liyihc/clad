from datetime import datetime
from enum import Enum
from pathlib import Path
from traceback import format_exc
from pydantic import BaseModel
import typer

app = typer.Typer(no_args_is_help=True)

from .models import models


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
        "-d", "--data"),
):
    params = Params.parse_file(config_file)
    model = params.get_model()
    output_folder = output_parent_folder / f"{params.dataset} {params.model} {datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_folder.mkdir(parents=True, exist_ok=True)

    try:
        getattr(model, "train", lambda x, y: 0)(data_folder / params.dataset.value, output_folder)
        model.test(data_folder / params.dataset.value, output_folder)
    except Exception:
        (output_folder / "error.txt").write_text(format_exc())
        from rich.console import Console
        console = Console()
        console.print_exception()



