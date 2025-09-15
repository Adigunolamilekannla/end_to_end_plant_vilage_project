from scr.Plant_Vilage.config.configuration import ConfigurationManager
from scr.Plant_Vilage.components.data_injection import DataInjection
from scr.Plant_Vilage import logger

class DataInjectionPipeLine:
    def __init__(self):
        pass
    def iniciate_data_injection(self):
        config = ConfigurationManager()
        data_injection_config=config.get_data_injection_config()
        import_data = DataInjection(config=data_injection_config)
        import_data.get_downloaded_data()



stage_name = "Data Injection Pipeline"

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>>>> Started {stage_name} <<<<<<<<<<<<")
        obj = DataInjectionPipeLine()
        ata_injection = obj.iniciate_data_injection()
        logger.info(f">>>>>>>>>>> {stage_name} Completed <<<<<<<<<<<<")
    except Exception as e:
        raise e    
        