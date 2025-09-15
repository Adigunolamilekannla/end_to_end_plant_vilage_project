import shutil
from pathlib import Path
from scr.Plant_Vilage.entity.config_entity import DataInjectionConfig
from scr.Plant_Vilage import logger
import os 


class DataInjection:

    def __init__(self,config:DataInjectionConfig):

        self.config = config



    def get_downloaded_data(self):

       # Source folder (where your dataset is now)
        source_folder = Path(self.config.file_location)
        # Destination folder (where you want to copy it)
        destination_folder = Path(self.config.main_data)
        # --- Option 1: Copy folder (keeps original) ---
        if not os.path.exists(self.config.main_data):
            shutil.copytree(source_folder, destination_folder / source_folder.name, dirs_exist_ok=True)
            logger.info(f"Copied folder: {source_folder} → {destination_folder}")
        else:
            logger.info(f"Data already exist at: {destination_folder}")