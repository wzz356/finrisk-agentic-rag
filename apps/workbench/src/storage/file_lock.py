"""Small cross-process lock shared by the web app and collection CLI."""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from time import monotonic, sleep
from typing import BinaryIO, Self


class FileLock:
    def __init__(self, path: Path, *, timeout: float = 30.0) -> None:
        self.path, self.timeout = path, timeout
        self._file: BinaryIO | None = None

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = monotonic() + self.timeout
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl = import_module("fcntl")
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._file = handle
                return self
            except OSError:
                if monotonic() >= deadline:
                    handle.close()
                    raise TimeoutError(f"任务正在运行: {self.path.name}") from None
                sleep(0.05)

    def __exit__(self, *exc: object) -> None:
        handle = self._file
        if handle is not None:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl = import_module("fcntl")
                fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
            self._file = None
