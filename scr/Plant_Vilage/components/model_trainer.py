import torch
import numpy as np
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.functional import F
import gc
from scr.Plant_Vilage import logger
from scr.Plant_Vilage.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self,config:ModelTrainerConfig):
        self.config = config
        self.device =  torch.device("cuda")

    def train_cnn_model(self):
        
        transform = transforms.Compose([
        transforms.Resize((224, 224)),                     # Resize images
        transforms.RandomHorizontalFlip(p=0.5),            # Randomly flip images
        transforms.RandomRotation(degrees=15),             # Random rotation
        #transforms.ColorJitter(brightness=0.2),  # Add jitter
        transforms.ToTensor(),                             # Convert to tensor [C, H, W] in [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5],          # Normalize with 0–1 range scaling
                            std=[0.5, 0.5, 0.5])
                ])


        train_dataset = datasets.ImageFolder(root=self.config.train_data_root,transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=self.config.shuffle, num_workers=self.config.num_workers)

        class BasicBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride=1, downsample=None):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                    stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                    stride=1, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.downsample = downsample

            def forward(self, x):
                identity = x

                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))

                if self.downsample:
                    identity = self.downsample(x)

                out += identity
                out = self.relu(out)

                return out

        class ResNetLike(nn.Module):
            def __init__(self, num_classes=1):
                super().__init__()
                self.in_channels = 64
                self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU(inplace=True)
                self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

                self.layer1 = self._make_layer(64, 2)
                self.layer2 = self._make_layer(128, 2, stride=2)
                self.layer3 = self._make_layer(256, 2, stride=2)
                self.layer4 = self._make_layer(512, 2, stride=2)

                self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(512, num_classes)

            def _make_layer(self, out_channels, blocks, stride=1):
                downsample = None
                if stride != 1 or self.in_channels != out_channels:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(out_channels),
                    )

                layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
                self.in_channels = out_channels
                for _ in range(1, blocks):
                    layers.append(BasicBlock(out_channels, out_channels))

                return nn.Sequential(*layers)

            def forward(self, x):
                x = self.relu(self.bn1(self.conv1(x)))  # [B, 64, H/2, W/2]
                x = self.pool(x)                        # [B, 64, H/4, W/4]
                x = self.layer1(x)                      # -> [B, 64, H/4, W/4]
                x = self.layer2(x)                      # -> [B, 128, H/8, W/8]
                x = self.layer3(x)                      # -> [B, 256, H/16, W/16]
                x = self.layer4(x)                      # -> [B, 512, H/32, W/32]
                x = self.global_pool(x)                 # -> [B, 512, 1, 1]
                x = torch.flatten(x, 1)                 # -> [B, 512]
                x = self.fc(x)                          # -> [B, num_classes]
                return x
            


        def get_model_optimizer():
            net = ResNetLike(num_classes=38)              # Make sure num_classes matches your dataset
            lossFun = nn.CrossEntropyLoss()               # For multi-class classification
            optimizer = torch.optim.Adam(net.parameters(), lr=self.config.learning_rate)
            return net, optimizer, lossFun
        

        def train_model(x_y_train_loader, device):
            
            torch.cuda.empty_cache()
            gc.collect()

            net, optimizer, lossFun = get_model_optimizer()
            net.to(device)

            num_epoch = self.config.num_epoch
            train_acc = np.zeros(num_epoch)
            train_loss = np.zeros(num_epoch)
            accumulation_step = 16

            for epoch in range(num_epoch):
                net.train()
                optimizer.zero_grad()
                batch_loss = []
                batch_acc = []

                for i, (X, y) in enumerate(x_y_train_loader):
                    # Flatten image input for FNN
                    X = X.to(device)  # (batch_size, 3*224*224)
                    y = y.to(device).long()  # CrossEntropyLoss expects LongTensor class indices

                    y_pred = net(X)
                    pred_labels = y_pred.argmax(dim=1)     # Get predicted class indices
                    acc = (pred_labels == y).float().mean().item()
                    batch_acc.append(acc)

                    loss = lossFun(y_pred, y)
                    loss = loss / accumulation_step
                    batch_loss.append(loss.item() * accumulation_step)  # Undo scaling for logging

                    loss.backward()

                    if (i + 1) % accumulation_step == 0 or (i + 1) == len(x_y_train_loader):
                        optimizer.step()
                        optimizer.zero_grad()

                train_acc[epoch] = np.mean(batch_acc)
                train_loss[epoch] = np.mean(batch_loss)

                

                logger.info(f"Epoch {epoch+1}/{num_epoch} | "
                    f"Train Loss: {train_loss[epoch]:.4f}, Train Acc: {train_acc[epoch]:.4f}")

            torch.cuda.empty_cache()
            gc.collect()

            torch.save(net.state_dict(),self.config.trained_model)
            logger.info(f"Plant Vilage Modeled SUcessfully the trained model is located at {self.config.trained_model}")


        train_model(x_y_train_loader=train_loader,device=self.device)





class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetLike(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # [B, 64, H/2, W/2]
        x = self.pool(x)                        # [B, 64, H/4, W/4]
        x = self.layer1(x)                      # -> [B, 64, H/4, W/4]
        x = self.layer2(x)                      # -> [B, 128, H/8, W/8]
        x = self.layer3(x)                      # -> [B, 256, H/16, W/16]
        x = self.layer4(x)                      # -> [B, 512, H/32, W/32]
        x = self.global_pool(x)                 # -> [B, 512, 1, 1]
        x = torch.flatten(x, 1)                 # -> [B, 512]
        x = self.fc(x)                          # -> [B, num_classes]
        return x
    


def get_model_optimizer():
    net = ResNetLike(num_classes=38)              # Make sure num_classes matches your dataset
    lossFun = nn.CrossEntropyLoss()               # For multi-class classification
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    return net, optimizer, lossFun

