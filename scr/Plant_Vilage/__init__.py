import os
import sys
import logging


logging_format = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s:]"

log_dir ="Logs"
log_filepath = os.path.join(log_dir,"running.log")
os.makedirs(log_dir,exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format=logging_format,

    handlers=[
        logging.FileHandler(log_filepath),# This path define where our logging should go
        logging.StreamHandler(sys.stdout) # Send log to our terminal console (sys.stdout just means "standard output")
    ]

)

logger = logging.getLogger("Plant_Vilage_Logger") #When you call getLogger("Data_Science_Logger"), your logger automatically uses those rules.