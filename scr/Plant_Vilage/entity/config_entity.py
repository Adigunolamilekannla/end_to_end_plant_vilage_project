from pathlib import Path
from dataclasses import dataclass


@dataclass
class DataInjectionConfig:
    dir_root: Path
    file_location: Path
    main_data:Path

    