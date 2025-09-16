from pathlib import Path
from dataclasses import dataclass


@dataclass
class DataInjectionConfig:
    dir_root: Path
    file_location: Path
    main_data:Path


@dataclass
class ModelTrainerConfig:
   dir_root: Path
   train_data_root: Path
   trained_model: Path
   num_epoch: int
   learning_rate: float
   num_classes: int
   batch_size: int
   num_workers: int
   shuffle: bool




@dataclass
class ModelEvaluationConfig:
    dir_root: Path
    test_data_root: Path
    load_trained_model: Path
    num_epoch: int
    learning_rate: float
    num_classes: int
    batch_size: int
    num_workers: int
    shuffle: bool
    classification_report_loc: Path
    mlflow_url: str
  
   