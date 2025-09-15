import os 
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s]: %(message)s:"
)

project_name = "Plant_Vilage"


list_of_files_and_directory = [
    ".github/workflows/",
    "app.py",
    "config_yaml/config.yaml",
    "templates/index.html",
    "templates/results.html",
    f"scr/{project_name}/utils/__init__.py",#
    f"scr/{project_name}/utils/common.py",
    f"scr/{project_name}/config/__init__.py",
    f"scr/{project_name}/config/configuration.py",
    f"scr/{project_name}/pipeline/__init__.py",
    f"scr/{project_name}/entity/__init__.py",
    f"scr/{project_name}/entity/config_entity.py",
    f"scr/{project_name}/constants/__init__.py",
    f"scr/{project_name}/__init__.py",
     f"scr/{project_name}/components/__init__.py",
    "params.yaml",
    "schema.yaml",
    "Dockerfile",
    "requirements.txt",
    "reseach/reseach.ipynb",
    "main.py",
    ".gitignore"
    
]


for f_and_d_Path in list_of_files_and_directory:
    filePath = Path(f_and_d_Path)
    filedir,filename = os.path.split(filePath)

    if os.path.exists(Path(filedir)):
        logging.info(f"Directory {filedir} not exists")
    else:  
        logging.info(f"Directory {filedir} already exists")

    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Sucessfully created directory {filedir} for file {filename}")


    if os.path.exists(Path(filedir)):
        logging.info(f"Directory {filedir} already exists")    

    if os.path.exists(Path(filedir)):
        logging.info(f"Directory {filedir} already exists")



    if (not os.path.exists(filePath)) or (os.path.exists(filePath) == 0)  :
        with  open(filePath,"w") as f:
            pass
            logging.info("Creating Empty Path")
