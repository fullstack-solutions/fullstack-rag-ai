import os
from typing import List
from .base import BaseSource
from ..config import SUPPORTED_EXTENSIONS, IGNORED_DIRS


class LocalSource(BaseSource):
    def __init__(self, path: str):
        self.path = path

    def load(self) -> List[str]:
        files = []

        for root, dirs, filenames in os.walk(self.path):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

            for f in filenames:
                if any(f.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                    files.append(os.path.join(root, f))

        return files