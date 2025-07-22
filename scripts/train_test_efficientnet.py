import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from pathlib import Path
import mlflow
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix


# Custom Dataset with Limited Debug Output
class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir) / 'final_dataset' / split
        self.transform = transform
        self.images = []
        self.labels = []

        print(f"Scanning directory for split {split}: {self.data_dir}")
        for item in self.data_dir.glob('*'):
            print(f"Found item: {item}")

        # Look for real and fake subdirectories
        for label, class_name in enumerate(['real', 'fake']):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                print(f"Scanning {class_name} directory: {class_dir}")
                for img_path in class_dir.glob('*.[jp][pn][gf]*'):  # Broad extension match
                    self.images.append(img_path)
                    self.labels.append(label)
            else:
                print(f"Directory {class_name} not found in {self.data_dir}")

        # Limit image list print to first 5 for brevity
        image_count = len(self.images)
        sample_images = self.images[:5] if image_count > 0 else []
        print(f"Found {image_count} images: {sample_images[:5] if sample_images else []}")
        print(f"Labels length: {len(self.labels)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = plt.imread(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label


# Data Transforms
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Model and Trainer
class DeepfakeTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
        self.train_losses = []
        self.val_accuracies = []

    def train(self, train_loader, val_loader, epochs=5):
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / len(train_loader)
            self.train_losses.append(epoch_loss)

            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_accuracy = 100 * correct / total
            self.val_accuracies.append(val_accuracy)
            self.scheduler.step(epoch_loss)
            print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    def test(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        test_accuracy = 100 * correct / total
        auc = roc_auc_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        print(f'Final Test Accuracy: {test_accuracy:.2f}%, AUC: {auc:.4f}')
        return test_accuracy, auc, cm


# Main Execution
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)  # Binary classification
    mlflow.set_tracking_uri("file:///kaggle/working/mlruns1")
    mlflow.set_experiment("Deepfake_Detection")
    with mlflow.start_run():
        DATA_DIR = Path("/kaggle/input/final-merged-dataset/")

        # Create datasets for each split
        train_dataset = DeepfakeDataset(DATA_DIR, split='train', transform=data_transforms)
        val_dataset = DeepfakeDataset(DATA_DIR, split='validation', transform=data_transforms)
        test_dataset = DeepfakeDataset(DATA_DIR, split='test', transform=data_transforms)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)

        trainer = DeepfakeTrainer(model, device)
        trainer.train(train_loader, val_loader)
        test_acc, auc, cm = trainer.test(test_loader)

        # Save model
        model_path_pth = Path("/kaggle/working/deepfake_model.pth")
        torch.save(model.state_dict(), model_path_pth)
        model_path_pickle = Path("/kaggle/working/deepfake_model.pkl")
        with open(model_path_pickle, 'wb') as f:
            pickle.dump(trainer.model, f)

        # Log metrics
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("auc", auc)
        mlflow.log_artifact(model_path_pth)
        mlflow.log_artifact(model_path_pickle)


if __name__ == "__main__":
    main()