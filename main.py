from scr.Plant_Vilage import logger
from scr.Plant_Vilage.pipeline.data_injection_pipeline import DataInjectionPipeLine
from scr.Plant_Vilage.pipeline.model_trainer_pipeline import ModelTrainerPipeline



stage_name = "Data Injection Pipeline"
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>>>> Started {stage_name} <<<<<<<<<<<<")
        obj = DataInjectionPipeLine()
        ata_injection = obj.iniciate_data_injection()
        logger.info(f">>>>>>>>>>> {stage_name} Completed <<<<<<<<<<<<")
    except Exception as e:
        raise e    


stage_name = "Model Training Stage"
try:
    logger.info(f">>>>>>>>>>> Started {stage_name} <<<<<<<<<<<<<<")
    model = ModelTrainerPipeline()
    model.train_model()
    logger.info(f">>>>>>>>>>> {stage_name} Completed <<<<<<<<<<<<<<")
except Exception as e:
     raise e    


