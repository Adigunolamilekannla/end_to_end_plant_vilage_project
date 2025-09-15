from scr.Plant_Vilage.constants import SCHEMA_FILE_PATH,PARAMS_FILE_PATH,CONFIG_FILE_PATH
import scr.Plant_Vilage.utils.common as common
create_diretory = common.create_directory 
read_yaml = common.read_yaml
from pathlib import Path
from scr.Plant_Vilage.entity.config_entity import DataInjectionConfig

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