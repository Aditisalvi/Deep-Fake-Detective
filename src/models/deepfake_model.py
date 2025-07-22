"""
DeepFake detection model implementation.

This module contains the model architecture, loading utilities,
and prediction functions for the DeepFake Detective system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Optional
import logging
from pathlib import Path

from ..config.config import config

# Setup logging
logger = logging.getLogger(__name__)

def create_model(num_classes: int = 2, pretrained: bool = True):
    """
    Create a DeepFake detection model (matching the original training structure).
    
    Args:
        num_classes (int): Number of output classes (default: 2 for real/fake)
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        torch.nn.Module: EfficientNet-B3 model for deepfake detection
    """
    # Load pre-trained EfficientNet-B3 (same structure as training script)
    if pretrained:
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
        model = models.efficientnet_b3(weights=weights)
    else:
        model = models.efficientnet_b3(weights=None)
    
    # Replace the classifier head
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    logger.info(f"Created EfficientNet-B3 model with {num_classes} classes")
    return model

class ModelLoader:
    """
    Utility class for loading and managing the DeepFake detection model.
    """
    
    @staticmethod
    def load_model(model_path: Optional[str] = None, device: str = "cpu") -> Optional[torch.nn.Module]:
        """
        Load a trained DeepFake detection model from file.
        
        Args:
            model_path (str, optional): Path to the model file
            device (str): Device to load the model on
            
        Returns:
            torch.nn.Module or None: Loaded model or None if loading failed
        """
        if model_path is None:
            model_path = config.model.model_path
        
        try:
            # Check if model file exists
            if not Path(model_path).exists():
                logger.error(f"Model file not found: {model_path}")
                return None
            
            # Create model architecture (same as training script)
            model = create_model(num_classes=config.model.num_classes)
            
            # Load state dict
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            
            logger.info(f"Successfully loaded model from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    @staticmethod
    def save_model(model: torch.nn.Module, model_path: Optional[str] = None) -> bool:
        """
        Save a trained model to file.
        
        Args:
            model (torch.nn.Module): Model to save
            model_path (str, optional): Path to save the model
            
        Returns:
            bool: True if successful, False otherwise
        """
        if model_path is None:
            model_path = config.model.model_path
        
        try:
            # Create directory if it doesn't exist
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model state dict
            torch.save(model.state_dict(), model_path)
            logger.info(f"Model saved successfully to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

class ModelPredictor:
    """
    Class for making predictions with the DeepFake detection model.
    """
    
    def __init__(self, model: torch.nn.Module):
        """
        Initialize the predictor with a model.
        
        Args:
            model (torch.nn.Module): Trained model for predictions
        """
        self.model = model
        self.model.eval()
    
    def predict(self, x: torch.Tensor) -> Tuple[int, float, float, float]:
        """
        Make a prediction on input tensor.
        
        Args:
            x (torch.Tensor): Preprocessed input tensor
            
        Returns:
            Tuple containing:
                - predicted_class (int): 0 for real, 1 for fake
                - confidence (float): Confidence of the prediction
                - real_prob (float): Probability of being real
                - fake_prob (float): Probability of being fake
        """
        with torch.no_grad():
            outputs = self.model(x)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get prediction (0: Real, 1: Fake)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Get both probabilities
            real_prob = probabilities[0][0].item()
            fake_prob = probabilities[0][1].item()
            
        return predicted_class, confidence, real_prob, fake_prob
    
    def predict_batch(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions on a batch of inputs.
        
        Args:
            x (torch.Tensor): Batch of preprocessed input tensors
            
        Returns:
            Tuple containing:
                - predictions (torch.Tensor): Predicted classes
                - probabilities (torch.Tensor): Prediction probabilities
        """
        with torch.no_grad():
            outputs = self.model(x)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
        return predictions, probabilities
