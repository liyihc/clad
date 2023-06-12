
from pathlib import Path
from pydantic import BaseModel
from .reader import DataReader


class BaseRunner(BaseModel):
    def train(self, data: DataReader, output_path: Path):
        pass

    def test(self, data: DataReader, output_path: Path):
        pass
