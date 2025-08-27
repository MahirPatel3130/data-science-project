from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionconfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
