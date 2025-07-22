"""
Gradient-based Class Activation Mapping (Grad-CAM) implementation.

This module provides visualization tools to understand what the deep learning
model focuses on when making predictions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, Tuple, Dict, List
import logging

from ..config.config import config

# Setup logging
logger = logging.getLogger(__name__)

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.
    
    This class provides functionality to generate attention heatmaps that show
    which parts of the input image the model focuses on for its predictions.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer_name: str):
        """
        Initialize GradCAM for a specific model and layer.
        
        Args:
            model: The neural network model
            target_layer_name: Name of the target layer for visualization
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks for gradient capture."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def full_backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach()
        
        # Find and register hooks on target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_layer = module
                break
        
        if target_layer is not None:
            forward_handle = target_layer.register_forward_hook(forward_hook)
            backward_handle = target_layer.register_full_backward_hook(full_backward_hook)
            self.hooks.extend([forward_handle, backward_handle])
            logger.debug(f"Registered hooks for layer: {self.target_layer_name}")
        else:
            raise ValueError(f"Layer {self.target_layer_name} not found in model")
    
    def cleanup_hooks(self):
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        logger.debug(f"Cleaned up hooks for layer: {self.target_layer_name}")
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Generate Class Activation Map for given input and class.
        
        Args:
            input_tensor: Input tensor for the model
            class_idx: Target class index (if None, uses predicted class)
            
        Returns:
            Tuple containing:
                - cam: Generated class activation map as numpy array
                - predicted_class: The class index used for CAM generation
        """
        try:
            # Ensure input requires gradients
            input_tensor.requires_grad_()\
            
            # Forward pass
            output = self.model(input_tensor)
            
            # Use predicted class if not specified
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            
            # Clear gradients and perform backward pass
            self.model.zero_grad()
            output[0, class_idx].backward(retain_graph=True)
            
            # Verify gradients and activations were captured
            if self.gradients is None or self.activations is None:
                raise ValueError("Failed to capture gradients or activations")
            
            # Generate CAM
            gradients = self.gradients[0]  # Remove batch dimension
            activations = self.activations[0]  # Remove batch dimension
            
            # Calculate weights via global average pooling of gradients
            weights = torch.mean(gradients, dim=(1, 2))
            
            # Weighted combination of activation maps
            cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
            for i, weight in enumerate(weights):
                cam += weight * activations[i]
            
            # Apply ReLU and normalize
            cam = F.relu(cam)
            if torch.max(cam) > 0:
                cam = cam / torch.max(cam)
            
            return cam.detach().numpy(), class_idx
            
        except Exception as e:
            logger.error(f"Error generating CAM for layer {self.target_layer_name}: {str(e)}")
            raise e
        finally:
            self.cleanup_hooks()

class VisualizationUtils:
    """
    Utility class for various visualization operations.
    """
    
    @staticmethod
    def apply_colormap_on_image(
        original_image: np.ndarray, 
        activation_map: np.ndarray, 
        colormap_name: str = "jet"
    ) -> np.ndarray:
        """
        Apply colormap overlay on original image.
        
        Args:
            original_image: Original image as numpy array
            activation_map: Activation/attention map
            colormap_name: Name of the colormap to use
            
        Returns:
            Superimposed image with heatmap overlay
        """
        # Ensure original image is in correct format
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image.convert('RGB'))
        
        # Handle different image formats
        if len(original_image.shape) == 3 and original_image.shape[2] == 4:
            original_image = original_image[:, :, :3]  # RGBA to RGB
        elif len(original_image.shape) == 2:
            original_image = np.stack([original_image] * 3, axis=-1)  # Grayscale to RGB
        
        # Resize activation map to match image dimensions
        activation_resized = cv2.resize(activation_map, (original_image.shape[1], original_image.shape[0]))
        
        # Apply colormap
        try:
            colormap = plt.colormaps[colormap_name]  # For matplotlib >= 3.7
        except AttributeError:
            colormap = plt.cm.get_cmap(colormap_name)  # Fallback for older versions
        
        # Generate heatmap
        heatmap = colormap(activation_resized)
        
        # Handle RGBA to RGB conversion if necessary
        if heatmap.shape[-1] == 4:
            heatmap = heatmap[:, :, :3]
        
        heatmap = np.uint8(255 * heatmap)
        
        # Blend images
        try:
            superimposed = (heatmap * config.visualization.heatmap_opacity + 
                          original_image * config.visualization.original_opacity)
            superimposed = np.uint8(superimposed)
        except ValueError:
            # Fallback: resize original image if shapes don't match
            original_resized = cv2.resize(original_image, (heatmap.shape[1], heatmap.shape[0]))
            superimposed = (heatmap * config.visualization.heatmap_opacity + 
                          original_resized * config.visualization.original_opacity)
            superimposed = np.uint8(superimposed)
        
        return superimposed
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            target_size: Target (width, height)
            
        Returns:
            Resized image
        """
        return cv2.resize(image, target_size)

class FeatureVisualizer:
    """
    Main class for generating feature visualizations from deep learning models.
    """
    
    def __init__(self, model: torch.nn.Module):
        """
        Initialize feature visualizer with a model.
        
        Args:
            model: The neural network model to visualize
        """
        self.model = model
        self.utils = VisualizationUtils()
    
    def visualize_features(
        self, 
        image: Image.Image, 
        target_layers: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, any]]:
        """
        Generate feature visualizations for multiple layers.
        
        Args:
            image: Input PIL image
            target_layers: List of layer names to visualize
            
        Returns:
            Dictionary containing visualizations for each layer
        """
        if target_layers is None:
            target_layers = config.visualization.target_layers
        
        from ..utils.preprocessing import ImagePreprocessor
        
        # Preprocess image
        preprocessor = ImagePreprocessor()
        processed_image = preprocessor.preprocess_image(image)
        
        # Prepare image for visualization
        if isinstance(image, Image.Image):
            image_np = np.array(image.convert('RGB'))
        else:
            image_np = image
        
        # Ensure RGB format
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]
        elif len(image_np.shape) == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        
        visualizations = {}
        
        for i, layer_name in enumerate(target_layers):
            try:
                # Create GradCAM instance for this layer
                grad_cam = GradCAM(self.model, layer_name)
                cam, predicted_class = grad_cam.generate_cam(processed_image.clone())
                
                # Apply colormap overlay
                visualization = self.utils.apply_colormap_on_image(
                    image_np.copy(), 
                    cam, 
                    config.visualization.colormap_name
                )
                
                layer_key = f"Layer_{i+1}_{layer_name.split('.')[-1]}"
                visualizations[layer_key] = {
                    'heatmap': cam,
                    'visualization': visualization,
                    'predicted_class': predicted_class,
                    'layer_name': layer_name
                }
                
                logger.info(f"Generated visualization for layer: {layer_name}")
                
            except Exception as e:
                logger.warning(f"Could not generate visualization for layer {layer_name}: {str(e)}")
                continue
        
        return visualizations
    
    def get_attention_statistics(self, visualizations: Dict[str, Dict[str, any]]) -> Dict[str, float]:
        """
        Calculate attention statistics across layers.
        
        Args:
            visualizations: Dictionary of layer visualizations
            
        Returns:
            Dictionary of attention statistics
        """
        stats = {}
        
        for layer_name, vis_data in visualizations.items():
            heatmap = vis_data['heatmap']
            stats[layer_name] = {
                'mean_attention': np.mean(heatmap),
                'max_attention': np.max(heatmap),
                'std_attention': np.std(heatmap),
                'attention_coverage': np.sum(heatmap > 0.5) / heatmap.size  # Percentage of high attention areas
            }
        
        return stats
