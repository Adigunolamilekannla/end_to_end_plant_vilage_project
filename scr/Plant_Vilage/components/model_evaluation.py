import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import mlflow
from urllib.parse import urlparse
from sklearn.metrics import classification_report
import pandas as pd
import os
from scr.Plant_Vilage.components.model_trainer import get_model_optimizer
from scr.Plant_Vilage.entity.config_entity import ModelEvaluationConfig


os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/leksman/end_to_end_plant_vilage_project.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "leksman"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "38aba5d8d599d1c5648f9c3d7aa353c04ce2c9f9" 







class ModelEvaluation:
    def __init__(self, config:ModelEvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate_model(self):
        # Transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        # Dataset + loader
        test_dataset = datasets.ImageFolder(
            root=self.config.test_data_root,
            transform=transform
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers
        )

        # Load model
        model, optimizer, lossFun = get_model_optimizer()
        model.load_state_dict(torch.load(self.config.load_trained_model, map_location=self.device, weights_only=True))
        model.to(self.device)
    

        # MLflow setup
        mlflow.set_registry_uri(self.config.mlflow_url)
        tracking_url_type = urlparse(mlflow.get_tracking_uri()).scheme

        # Evaluation
        correct, total = 0, 0
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        all_labels, all_preds = [], []

        with mlflow.start_run():
            model.eval()
            with torch.no_grad():
                for X_val, y_val in test_loader:
                    X_val = X_val.to(self.device)
                    y_val = y_val.to(self.device).long()

                    # Forward
                    y_pred = model(X_val)

                    # Loss
                    loss = criterion(y_pred, y_val)
                    total_loss += loss.item()

                    # Predictions
                    pred_labels = y_pred.argmax(dim=1)
                    correct += (pred_labels == y_val).sum().item()
                    total += y_val.size(0)

                    # Collect for classification report
                    all_labels.extend(y_val.cpu().numpy())
                    all_preds.extend(pred_labels.cpu().numpy())

            # Metrics
            accuracy = correct / total
            avg_loss = total_loss / len(test_loader)

            # Log metrics
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_loss", avg_loss)

            # Classification report
            report_dict = classification_report(
                all_labels,
                all_preds,
                target_names=test_dataset.classes,
                output_dict=True
            )

            # Save classification report as CSV
            report_df = pd.DataFrame(report_dict).transpose()
            report_csv_path =  self.config.classification_report_loc
            report_df.to_csv(report_csv_path, index=True)

            # Log report file to MLflow
            mlflow.log_artifact(report_csv_path)

            if tracking_url_type != "file":
                # Log model itself
                mlflow.pytorch.log_model(model, "cnn_model")
            else:
                mlflow.pytorch.log_model(model)

                 
