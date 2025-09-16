from scr.Plant_Vilage.components.model_trainer import ModelTrainer
from scr.Plant_Vilage.config.configuration import ConfigurationManager
from scr.Plant_Vilage import logger









class ModelTrainerPipeline:
    def __init__(self):
        pass

    def train_model(self):
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            train_cnn_model = ModelTrainer(config=model_trainer_config)
            train_cnn_model.train_cnn_model()


