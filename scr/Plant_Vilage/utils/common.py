import os 
import joblib
import json
from scr.Plant_Vilage import logger
from pathlib import Path
from box import ConfigBox
from ensure import ensure_annotations
import yaml



@ensure_annotations  
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns its content as a ConfigBox"""
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:   # ✅ safer than catching ValueError
                raise ValueError("YAML file is empty")
            logger.info(f"YAML file loaded successfully from: {path_to_yaml}")
            return ConfigBox(content)
    except Exception as e:
        logger.error(f"Error reading YAML file: {e}")
        raise e


@ensure_annotations
def create_directory(paths: list, verbose: bool = True):
    """Creates directories from a list of paths"""
    for path in paths:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):   
    """Saves a dictionary as a JSON file"""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at: {path}")    


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Loads a JSON file and returns a ConfigBox"""
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)    


def save_bin(data: object, path: Path):
    """Saves any Python object in binary format using joblib"""
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> object:
    """Loads a binary file using joblib"""
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data
