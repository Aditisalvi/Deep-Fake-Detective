"""
DeepFake Detective - Main Streamlit Application

A professional deepfake detection system with advanced feature visualization
and comprehensive analysis capabilities.

Author: Expert Data Scientist
Version: 2.0.0
"""

import streamlit as st
import torch
from PIL import Image
import numpy as np
import time
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add src to Python path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Import our custom modules
from src.config.config import config
from src.models.deepfake_model import ModelLoader, ModelPredictor
from src.utils.preprocessing import ImagePreprocessor, ImageEnhancer
from src.visualization.gradcam import FeatureVisualizer
from src.visualization.plots import PlotGenerator

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.logging.log_level),
    format=config.logging.log_format,
    handlers=[
        logging.FileHandler(config.logging.log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize global components
plot_generator = PlotGenerator()
image_preprocessor = ImagePreprocessor()
image_enhancer = ImageEnhancer()

# Configure Streamlit page
st.set_page_config(
    page_title=config.app.page_title,
    page_icon=config.app.page_icon,
    layout=config.app.layout,
    initial_sidebar_state=config.app.initial_sidebar_state
)

def load_custom_css():
    """Load custom CSS styling for the application."""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {{
        --primary-color: {config.app.primary_color};
        --secondary-color: {config.app.secondary_color};
        --accent-color: {config.app.accent_color};
        --success-color: {config.app.success_color};
        --warning-color: {config.app.warning_color};
        --danger-color: {config.app.danger_color};
    }}
    
    .main-header {{
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }}
    
    .prediction-card {{
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }}
    
    .fake-card {{
        background: linear-gradient(135deg, var(--danger-color) 0%, #ee5a24 100%);
    }}
    
    .real-card {{
        background: linear-gradient(135deg, var(--success-color) 0%, #54a0ff 100%);
    }}
    
    .metric-card {{
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin: 0.5rem 0;
    }}
    
    .sidebar-info {{
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid var(--primary-color);
        margin: 1rem 0;
    }}
    
    .stProgress .st-bo {{
        background-color: rgba(255, 255, 255, 0.1);
    }}
    
    .stProgress .st-bp {{
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    }}
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the deepfake detection model."""
    try:
        model = ModelLoader.load_model()
        if model is not None:
            logger.info("Model loaded successfully")
            return model
        else:
            logger.error("Failed to load model")
            return None
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Error loading model: {str(e)}")
        return None

def create_hero_section():
    """Create the main header section."""
    st.markdown("""
        <div class="main-header">
            <h1>🕵️ DeepFake Detective</h1>
            <p>Advanced AI-powered deepfake detection with explainable AI visualization</p>
        </div>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create the sidebar with controls and information."""
    with st.sidebar:
        st.markdown("### 🎛️ Controls")
        
        # Model information
        st.markdown(f"""
            <div class="sidebar-info">
                <h4>📊 Model Info</h4>
                <ul>
                    <li><b>Architecture:</b> {config.model.model_name.upper()}</li>
                    <li><b>Classes:</b> Real vs Fake</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        # Image enhancement controls
        st.markdown("### 🎨 Image Enhancement")
        brightness = st.slider(
            "Brightness", 
            *config.app.brightness_range, 
            1.0, 0.1,
            help="Adjust image brightness"
        )
        contrast = st.slider(
            "Contrast", 
            *config.app.contrast_range, 
            1.0, 0.1,
            help="Adjust image contrast"
        )
        saturation = st.slider(
            "Saturation", 
            *config.app.saturation_range, 
            1.0, 0.1,
            help="Adjust color saturation"
        )
        
        # Analysis settings
        st.markdown("### ⚙️ Analysis Settings")
        show_confidence_chart = st.checkbox("Show Confidence Chart", True)
        show_enhanced_image = st.checkbox("Show Enhanced Image", False)
        show_feature_visualization = st.checkbox("Show Feature Visualization", True)
        show_attention_plot = st.checkbox("Show Attention Analysis", True)
        
        # Information sections
        st.markdown("""
            <div class="sidebar-info">
                <h4>💡 Tips</h4>
                <ul>
                    <li>Use clear, well-lit face images</li>
                    <li>Front-facing photos work best</li>
                    <li>Avoid heavily compressed images</li>
                    <li>Single face images are optimal</li>
                </ul>
            </div>
            
            <div class="sidebar-info">
                <h4>🔬 Feature Visualization</h4>
                <ul>
                    <li><b>Grad-CAM:</b> Shows model attention</li>
                    <li><b>Heatmaps:</b> Red = high importance</li>
                    <li><b>Layers:</b> Different feature levels</li>
                    <li><b>Interpretability:</b> Understand decisions</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    return {
        'brightness': brightness,
        'contrast': contrast,
        'saturation': saturation,
        'show_confidence_chart': show_confidence_chart,
        'show_enhanced_image': show_enhanced_image,
        'show_feature_visualization': show_feature_visualization,
        'show_attention_plot': show_attention_plot
    }

def process_image(uploaded_file, settings):
    """Process uploaded image and return analysis results."""
    if uploaded_file is None:
        return None, None
    
    try:
        # Load image
        image = Image.open(uploaded_file)
        
        # Validate image
        if not image_preprocessor.validate_image(image):
            st.error("Invalid image format. Please upload a valid image file.")
            return None, None
        
        # Apply enhancements if requested
        if settings['show_enhanced_image']:
            analysis_image = image_enhancer.enhance_image(
                image,
                brightness=settings['brightness'],
                contrast=settings['contrast'],
                saturation=settings['saturation']
            )
        else:
            analysis_image = image
        
        return image, analysis_image
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        st.error(f"Error processing image: {str(e)}")
        return None, None

def make_prediction(model, image):
    """Make prediction on the given image."""
    try:
        predictor = ModelPredictor(model)
        processed_image = image_preprocessor.preprocess_image(image)
        
        predicted_class, confidence, real_prob, fake_prob = predictor.predict(processed_image)
        
        # Log prediction
        logger.info(f"Prediction made: Class={predicted_class}, Confidence={confidence:.3f}")
        
        return predicted_class, confidence, real_prob, fake_prob
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None, None

def display_prediction_results(predicted_class, confidence, real_prob, fake_prob, settings):
    """Display prediction results with visualizations."""
    
    # Main prediction display
    if predicted_class == 0:  # Real
        st.markdown(f"""
            <div class="prediction-card real-card">
                <h2>✅ REAL</h2>
                <h3>{confidence*100:.1f}% Confidence</h3>
                <p>This image appears to be authentic</p>
            </div>
        """, unsafe_allow_html=True)
    else:  # Fake
        st.markdown(f"""
            <div class="prediction-card fake-card">
                <h2>🚨 FAKE</h2>
                <h3>{confidence*100:.1f}% Confidence</h3>
                <p>This image appears to be manipulated</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Detailed metrics
    st.markdown("### 📊 Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Real Probability:**")
        st.progress(real_prob)
        st.write(f"{real_prob*100:.2f}%")
    
    with col2:
        st.markdown("**Fake Probability:**")
        st.progress(fake_prob)
        st.write(f"{fake_prob*100:.2f}%")
    
    # Confidence chart
    if settings['show_confidence_chart']:
        fig = plot_generator.create_confidence_chart(real_prob, fake_prob)
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk assessment
    #display_risk_assessment(confidence)

def display_risk_assessment(confidence):
    """Display risk assessment based on confidence."""
    st.markdown("### ⚠️ Risk Assessment")
    
    thresholds = config.app.risk_thresholds
    if confidence > thresholds["very_high"]:
        risk_level = "Very High"
        risk_color = config.app.danger_color
    elif confidence > thresholds["high"]:
        risk_level = "High"
        risk_color = config.app.warning_color
    elif confidence > thresholds["medium"]:
        risk_level = "Medium"
        risk_color = config.app.accent_color
    else:
        risk_level = "Low"
        risk_color = config.app.success_color
    
    st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {risk_color};">Confidence Level: {risk_level}</h4>
            <p>Based on model certainty: {confidence*100:.1f}%</p>
        </div>
    """, unsafe_allow_html=True)

def display_feature_visualization(model, analysis_image, settings):
    """Display feature visualization with tabs."""
    if not (settings['show_feature_visualization'] or settings['show_attention_plot']):
        return
    
    st.markdown("### 🔬 Feature Visualization")
    st.markdown("*Understanding what the AI model focuses on when making predictions*")
    
    with st.spinner('🧠 Generating feature visualizations...'):
        try:
            # Generate visualizations
            visualizer = FeatureVisualizer(model)
            visualizations = visualizer.visualize_features(analysis_image)
            
            if visualizations:
                # Create tabs
                tab_names = []
                if settings['show_feature_visualization']:
                    tab_names.append("🎯 Attention Heatmaps")
                if settings['show_attention_plot']:
                    tab_names.append("📊 Layer Analysis")
                
                tabs = st.tabs(tab_names)
                tab_idx = 0
                
                # Attention Heatmaps Tab
                if settings['show_feature_visualization']:
                    with tabs[tab_idx]:
                        st.markdown("**Red/warm colors indicate areas the model pays most attention to:**")
                        
                        cols = st.columns(config.visualization.visualization_grid_cols)
                        layer_names = list(visualizations.keys())
                        
                        for i, layer_name in enumerate(layer_names[:config.visualization.max_layers_display]):
                            with cols[i % config.visualization.visualization_grid_cols]:
                                st.markdown(f"**{layer_name}**")
                                vis_image = visualizations[layer_name]['visualization']
                                st.image(vis_image, caption=f"Attention Map - {layer_name}", use_container_width=True)
                        
                        # Interpretation guide
                        with st.expander("📚 Interpretation Guide"):
                            st.markdown("""
                            **Understanding the Heatmaps:**
                            - **🔴 Red/Hot Areas**: High model attention - these regions strongly influence the prediction
                            - **🔵 Blue/Cool Areas**: Low model attention - these regions have minimal impact
                            - **🟡 Yellow/Warm Areas**: Moderate attention - secondary important features
                            
                            **Layer-by-Layer Analysis:**
                            - **Early Layers**: Focus on edges, textures, and basic visual patterns
                            - **Middle Layers**: Detect shapes, patterns, and facial components
                            - **Later Layers**: Recognize complex facial features and semantic content
                            """)
                    tab_idx += 1
                
                # Layer Analysis Tab
                if settings['show_attention_plot']:
                    with tabs[tab_idx]:
                        attention_fig = plot_generator.create_attention_plot(visualizations)
                        st.plotly_chart(attention_fig, use_container_width=True)
                        
                        # Detailed interpretation
                        st.markdown("""
                        **📈 Understanding Layer-wise Attention:**
                        
                        This chart shows the average attention score across different network layers. 
                        Higher bars indicate layers that are more "active" or focused when analyzing this specific image.
                        
                        - **Layer 1**: Detects basic features like edges and textures
                        - **Layer 2**: Identifies patterns and simple shapes  
                        - **Layer 3**: Recognizes complex facial features
                        
                        **What this tells us:**
                        - Consistent high attention across layers suggests the model is confident
                        - Varying attention levels show different aspects are being analyzed
                        - Higher attention scores indicate the model is focusing more on specific regions
                        """)
            else:
                st.info("🚪 No feature visualizations could be generated. This may be due to model architecture constraints.")
                
        except Exception as e:
            logger.error(f"Error generating feature visualizations: {str(e)}")
            st.error(f"Error generating feature visualizations: {str(e)}")
            st.info("Feature visualization requires additional processing. The basic prediction still works correctly.")
            
            # Show explanation
            with st.expander("❓ Why might visualization fail?"):
                st.markdown("""
                Feature visualization can fail for several reasons:
                - **🏢 Model Architecture**: Some layers may not be compatible with GradCAM
                - **🔄 PyTorch Version**: Newer versions may have different hook behaviors  
                - **💾 Memory Constraints**: Visualization requires additional computation
                - **🏷️ Layer Names**: The target layers might not exist in this model variant
                
                💪 **The core deepfake detection still works perfectly!**
                """)

def create_about_section():
    """Create the about/information section."""
    with st.expander("ℹ️ About DeepFake Detective"):
        st.markdown("""
        ### How it works:
        
        **DeepFake Detective** uses a state-of-the-art EfficientNet-B3 neural network trained specifically 
        for deepfake detection. The model analyzes various aspects of facial images including:
        
        - **Facial inconsistencies**: Unnatural facial features or proportions
        - **Temporal artifacts**: Inconsistencies in lighting, shadows, and textures  
        - **Compression artifacts**: Signs of image manipulation or generation
        - **Biometric patterns**: Natural vs. artificial facial characteristics
        
        ### Feature Visualization:
        
        **Grad-CAM (Gradient-weighted Class Activation Mapping)** shows exactly what parts of the image 
        the model focuses on when making predictions:
        
        - **Attention Heatmaps**: Visual overlays showing which pixels influence the decision most
        - **Layer Analysis**: Different network layers focus on different types of features
        - **Interpretability**: Understand why the model classified an image as real or fake
        
        ### Model Performance:
        - **Architecture**: EfficientNet-B3 with custom classification head
        - **Training Data**: Diverse dataset of real and synthetic faces
        - **Output**: Binary classification (Real vs Fake) with confidence scores
        
        ### Limitations:
        - Performance may vary with image quality and resolution
        - New deepfake techniques may not be detected
        - Best results with clear, front-facing facial images
        - Multiple faces in one image may affect accuracy
        """)

def main():
    """Main application function."""
    # Load custom CSS
    load_custom_css()
    
    # Create hero section
    create_hero_section()
    
    # Load model
    model = load_model()
    if model is None:
        st.error("❌ Failed to load the model. Please ensure the model file exists and is accessible.")
        st.info("📍 Expected model location: `models/deepfake_model.pth`")
        return
    
    # Create sidebar and get settings
    settings = create_sidebar()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📸 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=config.data.supported_formats,
            help="Upload a face image to analyze for deepfake detection"
        )
        
        # Process and display images
        original_image, analysis_image = process_image(uploaded_file, settings)
        
        if original_image is not None:
            st.image(original_image, caption="Original Image", use_container_width=True)
            
            if settings['show_enhanced_image'] and analysis_image is not original_image:
                st.image(analysis_image, caption="Enhanced Image", use_container_width=True)
    
    with col2:
        if analysis_image is not None:
            st.markdown("### 🔍 Analysis Results")
            
            # Make prediction
            with st.spinner('🧠 Analyzing image...'):
                time.sleep(0.5)  # Small delay for better UX
                predicted_class, confidence, real_prob, fake_prob = make_prediction(model, analysis_image)
            
            if predicted_class is not None:
                # Display results
                display_prediction_results(predicted_class, confidence, real_prob, fake_prob, settings)
                
                # Feature visualization
                display_feature_visualization(model, analysis_image, settings)
    
    # About section
    create_about_section()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #666;">
            <p>🔬 Built with 💜 by Aditi Salvi</p>
            <p> My Notebook - https://www.kaggle.com/code/aditisalvi013/deep-fake-project</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
