from scr.Plant_Vilage.components.model_evaluation import ModelEvaluation
from scr.Plant_Vilage.config.configuration import ConfigurationManager

class ModelEvaluationPipeline:
    def __init__(self):
        pass
    def evaluate_model(self):
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            model_eval = ModelEvaluation(config=model_evaluation_config)
            model_eval.evaluate_model()


            