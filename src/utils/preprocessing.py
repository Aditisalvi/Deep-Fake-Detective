"""
Image preprocessing utilities for the DeepFake Detective project.

This module contains functions for image loading, preprocessing,
and enhancement operations.
"""

import torch
from torchvision import transforms
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from typing import Union, Tuple, Optional
import logging

from ..config.config import config

# Setup logging
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Class for handling image preprocessing operations.
    """
    
    def __init__(self):
        """Initialize the preprocessor with default transforms."""
        self.transforms = self._get_transforms()
    
    def _get_transforms(self) -> transforms.Compose:
        """
        Get image preprocessing transforms.
        
        Returns:
            Composed transforms for image preprocessing
        """
        return transforms.Compose([
            transforms.Resize(config.model.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.data.image_mean,
                std=config.data.image_std
            )
        ])
    
    def preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Preprocessed tensor ready for model input
        """
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                logger.debug(f"Converted image from {image.mode} to RGB")
            
            # Apply transforms
            processed = self.transforms(image).unsqueeze(0)
            logger.debug(f"Preprocessed image shape: {processed.shape}")
            
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise e
    
    def validate_image(self, image: Union[Image.Image, np.ndarray]) -> bool:
        """
        Validate image format and properties.
        
        Args:
            image: Image to validate
            
        Returns:
            True if image is valid, False otherwise
        """
        try:
            if isinstance(image, np.ndarray):
                # Check array properties
                if len(image.shape) not in [2, 3]:
                    logger.warning("Image must be 2D (grayscale) or 3D (color)")
                    return False
                
                if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
                    logger.warning("Color image must have 1, 3, or 4 channels")
                    return False
                    
            elif isinstance(image, Image.Image):
                # Check PIL image properties
                if image.size[0] == 0 or image.size[1] == 0:
                    logger.warning("Image has zero dimensions")
                    return False
                    
                # Check if image mode is supported
                supported_modes = ['L', 'RGB', 'RGBA', 'P']
                if image.mode not in supported_modes:
                    logger.warning(f"Unsupported image mode: {image.mode}")
                    return False
                    
            else:
                logger.warning("Image must be PIL Image or numpy array")
                return False
            
            logger.debug("Image validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating image: {str(e)}")
            return False

class ImageEnhancer:
    """
    Class for image enhancement operations.
    """
    
    @staticmethod
    def enhance_image(
        image: Image.Image,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        sharpness: float = 1.0
    ) -> Image.Image:
        """
        Apply various enhancements to an image.
        
        Args:
            image: PIL Image to enhance
            brightness: Brightness factor (1.0 = no change)
            contrast: Contrast factor (1.0 = no change)
            saturation: Color saturation factor (1.0 = no change)
            sharpness: Sharpness factor (1.0 = no change)
            
        Returns:
            Enhanced PIL Image
        """
        try:
            enhanced = image.copy()
            
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(enhanced)
                enhanced = enhancer.enhance(brightness)
                logger.debug(f"Applied brightness: {brightness}")
            
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(contrast)
                logger.debug(f"Applied contrast: {contrast}")
            
            if saturation != 1.0:
                enhancer = ImageEnhance.Color(enhanced)
                enhanced = enhancer.enhance(saturation)
                logger.debug(f"Applied saturation: {saturation}")
            
            if sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(enhanced)
                enhanced = enhancer.enhance(sharpness)
                logger.debug(f"Applied sharpness: {sharpness}")
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing image: {str(e)}")
            return image  # Return original image if enhancement fails
    
    @staticmethod
    def resize_image(
        image: Union[Image.Image, np.ndarray],
        size: Tuple[int, int],
        maintain_aspect_ratio: bool = True
    ) -> Union[Image.Image, np.ndarray]:
        """
        Resize image to specified dimensions.
        
        Args:
            image: Image to resize
            size: Target size (width, height)
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        try:
            if isinstance(image, Image.Image):
                if maintain_aspect_ratio:
                    image.thumbnail(size, Image.Resampling.LANCZOS)
                    return image
                else:
                    return image.resize(size, Image.Resampling.LANCZOS)
            
            elif isinstance(image, np.ndarray):
                if maintain_aspect_ratio:
                    h, w = image.shape[:2]
                    target_w, target_h = size
                    
                    # Calculate scaling factor
                    scale = min(target_w / w, target_h / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    
                    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                else:
                    return cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
            
        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            return image

class ImageUtils:
    """
    Utility functions for image operations.
    """
    
    @staticmethod
    def convert_image_format(
        image: Union[Image.Image, np.ndarray],
        target_format: str = "RGB"
    ) -> Union[Image.Image, np.ndarray]:
        """
        Convert image to specified format.
        
        Args:
            image: Input image
            target_format: Target format (RGB, BGR, etc.)
            
        Returns:
            Converted image
        """
        try:
            if isinstance(image, Image.Image):
                return image.convert(target_format)
            
            elif isinstance(image, np.ndarray):
                if target_format == "RGB" and len(image.shape) == 3:
                    if image.shape[2] == 4:  # RGBA to RGB
                        return image[:, :, :3]
                    elif image.shape[2] == 3:  # Assume BGR to RGB
                        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif target_format == "BGR" and len(image.shape) == 3:
                    if image.shape[2] == 3:  # RGB to BGR
                        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                elif target_format == "GRAY":
                    if len(image.shape) == 3:
                        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                
                return image
                
        except Exception as e:
            logger.error(f"Error converting image format: {str(e)}")
            return image
    
    @staticmethod
    def get_image_info(image: Union[Image.Image, np.ndarray]) -> dict:
        """
        Get comprehensive information about an image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary containing image information
        """
        info = {}
        
        try:
            if isinstance(image, Image.Image):
                info.update({
                    'type': 'PIL',
                    'mode': image.mode,
                    'size': image.size,
                    'format': image.format,
                    'has_transparency': image.mode in ['RGBA', 'LA', 'P']
                })
                
            elif isinstance(image, np.ndarray):
                info.update({
                    'type': 'numpy',
                    'shape': image.shape,
                    'dtype': str(image.dtype),
                    'min_value': float(np.min(image)),
                    'max_value': float(np.max(image)),
                    'mean_value': float(np.mean(image))
                })
                
                # Determine number of channels
                if len(image.shape) == 2:
                    info['channels'] = 1
                elif len(image.shape) == 3:
                    info['channels'] = image.shape[2]
                
            logger.debug(f"Image info: {info}")
            
        except Exception as e:
            logger.error(f"Error getting image info: {str(e)}")
            info['error'] = str(e)
        
        return info
    
    @staticmethod
    def save_image(
        image: Union[Image.Image, np.ndarray],
        filepath: str,
        quality: int = 95
    ) -> bool:
        """
        Save image to file.
        
        Args:
            image: Image to save
            filepath: Output file path
            quality: JPEG quality (0-100)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if isinstance(image, Image.Image):
                image.save(filepath, quality=quality, optimize=True)
            elif isinstance(image, np.ndarray):
                # Convert RGB to BGR for OpenCV
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(filepath, image_bgr)
                else:
                    cv2.imwrite(filepath, image)
            
            logger.info(f"Image saved to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            return False
