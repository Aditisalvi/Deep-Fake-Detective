"""
Configuration settings for the DeepFake Detective project.

This module contains all configuration parameters for the project including
model settings, paths, visualization parameters, and application settings.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

@dataclass
class ModelConfig:
    """Configuration for model-related settings."""
    model_name: str = "efficientnet_b3"
    num_classes: int = 2
    input_size: tuple = (224, 224)
    model_path: str = str(PROJECT_ROOT / "models" / "deepfake_model.pth")
    device: str = "cpu"  # Will be auto-detected
    
    # Model architecture settings
    pretrained_weights: str = "IMAGENET1K_V1"
    
@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    # GradCAM settings
    target_layers: List[str] = field(default_factory=lambda: [
        'features.2.0',  # Early-mid features
        'features.4.0',  # Mid-level features  
        'features.6.0',  # Higher-level features
    ])
    
    # Colormap settings
    colormap_name: str = "jet"
    heatmap_opacity: float = 0.4
    original_opacity: float = 0.6
    
    # Display settings
    max_layers_display: int = 4
    visualization_grid_cols: int = 2

@dataclass
class DataConfig:
    """Configuration for data-related settings."""
    # Image preprocessing
    image_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    image_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Supported file types
    supported_formats: List[str] = field(default_factory=lambda: ['png', 'jpg', 'jpeg'])
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    
    # Data paths
    raw_data_path: str = str(PROJECT_ROOT / "data" / "raw")
    processed_data_path: str = str(PROJECT_ROOT / "data" / "processed")

@dataclass
class AppConfig:
    """Configuration for Streamlit application."""
    page_title: str = "DeepFake Detective 🕵️"
    page_icon: str = "🕵️"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    
    # Theme colors
    primary_color: str = "#667eea"
    secondary_color: str = "#764ba2"
    accent_color: str = "#f093fb"
    success_color: str = "#4ade80"
    warning_color: str = "#f59e0b"
    danger_color: str = "#ef4444"
    
    # Enhancement controls
    brightness_range: tuple = (0.5, 2.0)
    contrast_range: tuple = (0.5, 2.0)
    saturation_range: tuple = (0.5, 2.0)
    
    # Risk assessment thresholds
    risk_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "very_high": 0.9,
        "high": 0.8,
        "medium": 0.7
    })

@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_level: str = "INFO"
    log_file: str = str(PROJECT_ROOT / "logs" / "app.log")
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

@dataclass
class Config:
    """Main configuration class that combines all config sections."""
    model: ModelConfig = field(default_factory=ModelConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    app: AppConfig = field(default_factory=AppConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Create necessary directories
        os.makedirs(os.path.dirname(self.logging.log_file), exist_ok=True)
        os.makedirs(self.data.raw_data_path, exist_ok=True)
        os.makedirs(self.data.processed_data_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.model.model_path), exist_ok=True)

# Global configuration instance
config = Config()

# Environment-specific overrides
if os.getenv("ENVIRONMENT") == "production":
    config.logging.log_level = "WARNING"
elif os.getenv("ENVIRONMENT") == "development":
    config.logging.log_level = "DEBUG"
