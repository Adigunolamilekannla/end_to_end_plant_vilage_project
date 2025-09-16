from scr.Plant_Vilage.constants import SCHEMA_FILE_PATH,PARAMS_FILE_PATH,CONFIG_FILE_PATH
import scr.Plant_Vilage.utils.common as common
create_diretory = common.create_directory 
read_yaml = common.read_yaml
from pathlib import Path
from scr.Plant_Vilage.entity.config_entity import DataInjectionConfig
from scr.Plant_Vilage.entity.config_entity import ModelTrainerConfig
from scr.Plant_Vilage.entity.config_entity import ModelEvaluationConfig


class ConfigurationManager:
    def __init__(
        self,
        config_file_path: Path = CONFIG_FILE_PATH,
        schema_file_path: Path = SCHEMA_FILE_PATH,
        params_file_path: Path = PARAMS_FILE_PATH
    ):
       
        self.config = read_yaml(config_file_path)
        self.schema = read_yaml(schema_file_path)
        self.params = read_yaml(params_file_path)
        create_diretory([self.config.artifacts_root])

    def get_data_injection_config(self) -> DataInjectionConfig:
        config = self.config.data_injection
        create_diretory([config.dir_root])

        data_injection_config = DataInjectionConfig(
            dir_root=config.dir_root,
            file_location=config.file_location,
            main_data=config.main_data
        )
        return data_injection_config
    

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.model_params
        create_diretory([config.dir_root])

        model_trainer_config = ModelTrainerConfig(
            dir_root= config.dir_root,
            train_data_root=config.train_data_root,
            trained_model=config.trained_model,
            num_epoch = params.num_epoch,
            learning_rate=params.learning_rate,
            num_classes=params.num_classes,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
            shuffle=params.shuffle
    

        )

        return model_trainer_config
    


    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.model_params
        create_diretory([config.dir_root])

        model_evaluation_config = ModelEvaluationConfig(
            dir_root= config.dir_root,
            load_trained_model=config.load_trained_model,
            num_epoch = params.num_epoch,
            learning_rate=params.learning_rate,
            num_classes=params.num_classes,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
            shuffle=params.shuffle,
            classification_report_loc= config.classification_report_loc,
            test_data_root=config.test_data_root,
            mlflow_url=config.mlflow_url
            
    

        )

        return model_evaluation_config
