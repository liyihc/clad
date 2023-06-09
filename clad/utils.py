from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Collection, TypeVar, Union

from rich.live import Live
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn
from rich.traceback import install
install(show_locals=False)

V = TypeVar('V')

@dataclass
class Bar:
    progress: Progress
    job_id: int

    def iter(
            self,
            it: Iterable[V],
            description: str = "") -> Iterator[V]:
        try:
            total = len(it)
        except:
            total = -1

        self.progress.reset(self.job_id)
        self.progress.update(
            self.job_id,
            total=total if total > 0 else 3,
            description=description)
        if total < 0:
            self.progress.advance(self.job_id)
        for i in it:
            yield i
            if total > 0:
                self.progress.advance(self.job_id)

    def range(
            self,
            *args,
            description=""):
        for arg in args:
            assert not isinstance(arg, str), f"'{arg}' is not int, it's maybe description"
        return self.iter(
            range(*args),
            description=description)

    def update(self, description: str = None):
        self.progress.update(self.job_id, description=description)


@contextmanager
def progress_bar(num: int = 1, refresh_hz=1):
    job_progress = Progress(
        "{task.description}",
        TimeElapsedColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn()
    )

    bars = tuple(Bar(job_progress, job_progress.add_task(""))
                 for _ in range(num))

    with Live(job_progress, refresh_per_second=refresh_hz):
        yield bars
