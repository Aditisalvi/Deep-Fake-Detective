import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import io
import time
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.nn import functional as F
from typing import List, Tuple

# Configure the page
st.set_page_config(
    page_title="DeepFake Detective 🕵️",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for elegant styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .fake-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    }
    
    .real-card {
        background: linear-gradient(135deg, #00d2d3 0%, #54a0ff 100%);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin: 0.5rem 0;
    }
    
    .stProgress .st-bo {
        background-color: rgba(255, 255, 255, 0.1);
    }
    
    .stProgress .st-bp {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .sidebar-info {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Model class definition - Match the training script structure
def create_model(num_classes=2):
    """Create the same model structure as used in training"""
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM)"""
    
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []  # Store hook handles for cleanup
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def full_backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach()
            
        # Find target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_layer = module
                break
                
        if target_layer is not None:
            # Register hooks and store handles for cleanup
            forward_handle = target_layer.register_forward_hook(forward_hook)
            backward_handle = target_layer.register_full_backward_hook(full_backward_hook)
            self.hooks.extend([forward_handle, backward_handle])
        else:
            raise ValueError(f"Layer {self.target_layer_name} not found in model")
    
    def cleanup_hooks(self):
        """Remove registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate_cam(self, input_tensor, class_idx=None):
        """Generate CAM for given input and class"""
        try:
            # Ensure input requires grad
            input_tensor.requires_grad_()
            
            # Forward pass
            output = self.model(input_tensor)
            
            # If class_idx is None, use the predicted class
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
                
            # Zero gradients
            self.model.zero_grad()
            
            # Backward pass
            output[0, class_idx].backward(retain_graph=True)
            
            # Check if gradients and activations were captured
            if self.gradients is None or self.activations is None:
                raise ValueError("Failed to capture gradients or activations")
            
            # Get gradients and activations
            gradients = self.gradients[0]  # Remove batch dimension
            activations = self.activations[0]  # Remove batch dimension
            
            # Calculate weights (global average pooling of gradients)
            weights = torch.mean(gradients, dim=(1, 2))
            
            # Generate CAM
            cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
            for i, w in enumerate(weights):
                cam += w * activations[i]
                
            # Apply ReLU
            cam = F.relu(cam)
            
            # Normalize
            if torch.max(cam) > 0:
                cam = cam / torch.max(cam)
            
            return cam.detach().numpy(), class_idx
            
        except Exception as e:
            self.cleanup_hooks()  # Clean up hooks on error
            raise e
        finally:
            # Clean up hooks after use
            self.cleanup_hooks()

def apply_colormap_on_image(org_im, activation, colormap_name='jet'):
    """Apply colormap on image"""
    # Ensure org_im is numpy array
    if isinstance(org_im, Image.Image):
        org_im = np.array(org_im)
    
    # Ensure org_im is RGB (3 channels)
    if len(org_im.shape) == 3 and org_im.shape[2] == 4:  # RGBA to RGB
        org_im = org_im[:, :, :3]
    elif len(org_im.shape) == 2:  # Grayscale to RGB
        org_im = np.stack([org_im] * 3, axis=-1)
    
    # Resize activation to match image size
    activation_resized = cv2.resize(activation, (org_im.shape[1], org_im.shape[0]))
    
    # Apply colormap (fix matplotlib deprecation)
    try:
        colormap = plt.colormaps[colormap_name]  # New way for matplotlib >= 3.7
    except AttributeError:
        colormap = cm.get_cmap(colormap_name)  # Fallback for older versions
    
    # Apply colormap and ensure we get the right format
    heatmap = colormap(activation_resized)
    
    # Convert to uint8 and handle RGBA -> RGB conversion
    if heatmap.shape[-1] == 4:  # RGBA
        heatmap_rgb = heatmap[:, :, :3]  # Drop alpha channel
    else:  # RGB
        heatmap_rgb = heatmap
    
    heatmap_rgb = np.uint8(255 * heatmap_rgb)
    
    # Ensure both images have same number of channels
    if heatmap_rgb.shape[-1] != org_im.shape[-1]:
        if heatmap_rgb.shape[-1] == 3 and len(org_im.shape) == 2:
            org_im = np.stack([org_im] * 3, axis=-1)
        elif len(heatmap_rgb.shape) == 2 and org_im.shape[-1] == 3:
            heatmap_rgb = np.stack([heatmap_rgb] * 3, axis=-1)
    
    # Superimpose the heatmap on original image
    try:
        superimposed_img = heatmap_rgb * 0.4 + org_im * 0.6
        superimposed_img = np.uint8(superimposed_img)
    except ValueError as e:
        # If shapes still don't match, resize org_im to match heatmap
        org_im_resized = cv2.resize(org_im, (heatmap_rgb.shape[1], heatmap_rgb.shape[0]))
        superimposed_img = heatmap_rgb * 0.4 + org_im_resized * 0.6
        superimposed_img = np.uint8(superimposed_img)
    
    return superimposed_img

def visualize_features(model, image, target_layers=None):
    """Generate feature visualizations for multiple layers"""
    if target_layers is None:
        # Simpler, more reliable layers for EfficientNet-B3
        target_layers = [
            'features.2.0',  # Early-mid features
            'features.4.0',  # Mid-level features  
            'features.6.0',  # Higher-level features
        ]
    
    processed_image = preprocess_image(image)
    visualizations = {}
    
    # Convert image to proper numpy format once
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert('RGB'))  # Ensure RGB
    else:
        image_np = image
        
    # Ensure image is RGB (3 channels)
    if len(image_np.shape) == 3 and image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]  # Convert RGBA to RGB
    elif len(image_np.shape) == 2:
        image_np = np.stack([image_np] * 3, axis=-1)  # Convert grayscale to RGB
    
    for i, layer_name in enumerate(target_layers):
        try:
            # Create a fresh GradCAM instance for each layer
            grad_cam = GradCAM(model, layer_name)
            cam, predicted_class = grad_cam.generate_cam(processed_image.clone())
            
            # Apply colormap with proper image format
            visualization = apply_colormap_on_image(image_np.copy(), cam)
            
            visualizations[f"Layer_{i+1}_{layer_name.split('.')[-1]}"] = {
                'heatmap': cam,
                'visualization': visualization,
                'predicted_class': predicted_class
            }
            
        except Exception as e:
            # More detailed error information
            error_msg = f"Layer {layer_name}: {str(e)}"
            if "broadcast" in str(e):
                error_msg += f" (Image shape: {image_np.shape})"
            st.warning(f"Could not generate visualization for {error_msg}")
            continue
            
    return visualizations

def create_attention_plot(attention_maps):
    """Create an interactive plot showing attention across different layers"""
    fig = go.Figure()
    
    layer_names = list(attention_maps.keys())
    attention_scores = [np.mean(attention_maps[layer]['heatmap']) for layer in layer_names]
    
    fig.add_trace(go.Bar(
        x=[f"Layer {i+1}" for i in range(len(layer_names))],
        y=attention_scores,
        marker_color=['#667eea', '#764ba2', '#f093fb', '#ff6b6b'][:len(layer_names)],
        text=[f'{score:.3f}' for score in attention_scores],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Average Attention: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Model Attention Across Layers",
        xaxis_title="Network Layers",
        yaxis_title="Average Attention Score",
        template="plotly_dark",
        height=400,
        left=2,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

@st.cache_resource
def load_model():
    """Load the deepfake detection model"""
    try:
        # Create model using the same structure as training script
        model = create_model(num_classes=2)
        
        # Load the state dict
        state_dict = torch.load('deepfake_model.pth', map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_transforms():
    """Get image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def preprocess_image(image):
    """Preprocess image for model input"""
    transform = get_transforms()
    
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    processed = transform(image).unsqueeze(0)
    return processed

def predict_image(model, image):
    """Make prediction on image"""
    processed_image = preprocess_image(image)
    
    with torch.no_grad():
        outputs = model(processed_image)
        probabilities = F.softmax(outputs, dim=1)
        
        # Get prediction (0: Real, 1: Fake)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # Get both probabilities
        real_prob = probabilities[0][0].item()
        fake_prob = probabilities[0][1].item()
        
    return predicted_class, confidence, real_prob, fake_prob

def create_confidence_chart(real_prob, fake_prob):
    """Create confidence visualization"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Real', 'Fake'],
        y=[real_prob * 100, fake_prob * 100],
        marker_color=['#00d2d3', '#ff6b6b'],
        text=[f'{real_prob*100:.1f}%', f'{fake_prob*100:.1f}%'],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Prediction Confidence",
        yaxis_title="Confidence (%)",
        xaxis_title="Classification",
        template="plotly_dark",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def enhance_image(image, brightness=1.0, contrast=1.0, saturation=1.0):
    """Apply image enhancements"""
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
    
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
    
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation)
    
    return image

def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🕵️ DeepFake Detective</h1>
            <p>Advanced AI-powered deepfake detection using EfficientNet-B3</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("❌ Failed to load the model. Please ensure 'deepfake_model.pth' is in the current directory.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Controls")
        
        st.markdown("""
            <div class="sidebar-info">
                <h4>📊 Model Info</h4>
                <ul>
                    <li><b>Architecture:</b> EfficientNet-B3</li>
                    <li><b>Classes:</b> Real vs Fake</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        # Image enhancement controls
        st.markdown("### 🎨 Image Enhancement")
        brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
        contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
        saturation = st.slider("Saturation", 0.5, 2.0, 1.0, 0.1)
        
        # Analysis settings
        st.markdown("### ⚙️ Analysis Settings")
        show_confidence_chart = st.checkbox("Show Confidence Chart", True)
        show_enhanced_image = st.checkbox("Show Enhanced Image", False)
        show_feature_visualization = st.checkbox("Show Feature Visualization", True)
        show_attention_plot = st.checkbox("Show Attention Plot", True)
        
        st.markdown("""
            <div class="sidebar-info">
                <h4>💡 Tips</h4>
                <ul>
                    <li>Use clear, well-lit face images</li>
                    <li>Front-facing photos work best</li>
                    <li>Avoid heavily compressed images</li>
                    <li>Multiple faces may affect accuracy</li>
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
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📸 Upload Image")
        
        # File uploader with drag and drop styling
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a face image to analyze for deepfake detection"
        )
        
        if uploaded_file is not None:
            # Load and display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_container_width=True)
            
            # Show enhanced image if requested
            if show_enhanced_image:
                enhanced_image = enhance_image(image, brightness, contrast, saturation)
                st.image(enhanced_image, caption="Enhanced Image", use_container_width=True)
                analysis_image = enhanced_image
            else:
                analysis_image = image
    
    with col2:
        if uploaded_file is not None:
            st.markdown("### 🔍 Analysis Results")
            
            # Show loading animation
            with st.spinner('🧠 Analyzing image...'):
                time.sleep(0.5)  # Small delay for better UX
                predicted_class, confidence, real_prob, fake_prob = predict_image(model, analysis_image)
            
            # Display prediction
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
            
            # Probability bars
            st.markdown("**Real Probability:**")
            st.progress(real_prob)
            st.write(f"{real_prob*100:.2f}%")
            
            st.markdown("**Fake Probability:**")
            st.progress(fake_prob)
            st.write(f"{fake_prob*100:.2f}%")
            
            # Confidence chart
            if show_confidence_chart:
                fig = create_confidence_chart(real_prob, fake_prob)
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature Visualization Section with Tabs
            if show_feature_visualization or show_attention_plot:
                st.markdown("### 🔬 Feature Visualization")
                st.markdown("*Understanding what the AI model focuses on when making predictions*")
                
                with st.spinner('🧠 Generating feature visualizations...'):
                    try:
                        # Generate feature visualizations
                        visualizations = visualize_features(model, analysis_image)
                        
                        if visualizations:
                            # Debug information (can be removed later)
                            if isinstance(analysis_image, Image.Image):
                                img_info = f"PIL Image - Size: {analysis_image.size}, Mode: {analysis_image.mode}"
                            else:
                                img_info = f"NumPy Array - Shape: {analysis_image.shape}"
                            
                            # Create tabs for better organization
                            tab_names = []
                            if show_feature_visualization:
                                tab_names.append("🎯 Attention Heatmaps")
                            if show_attention_plot:
                                tab_names.append("📊 Layer Analysis")
                            
                            tabs = st.tabs(tab_names)
                            tab_idx = 0
                            
                            # Attention Heatmaps Tab
                            if show_feature_visualization:
                                with tabs[tab_idx]:
                                    st.markdown("**Red/warm colors indicate areas the model pays most attention to:**")
                                    
                                    # Create columns for different layer visualizations
                                    cols = st.columns(2)
                                    layer_names = list(visualizations.keys())
                                    
                                    for i, layer_name in enumerate(layer_names[:4]):  # Show up to 4 layers
                                        with cols[i % 2]:
                                            st.markdown(f"**{layer_name}**")
                                            # Display the visualization directly (it's already in RGB)
                                            vis_image = visualizations[layer_name]['visualization']
                                            st.image(vis_image, caption=f"Attention Map - {layer_name}", use_container_width=True)
                                    
                                    # Add interpretation guide
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
                            if show_attention_plot:
                                with tabs[tab_idx]:
                                    attention_fig = create_attention_plot(visualizations)
                                    st.plotly_chart(attention_fig, use_container_width=True)
                                    
                                    # Add detailed interpretation
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
                        st.error(f"Error generating feature visualizations: {str(e)}")
                        st.info("Feature visualization requires additional processing. The basic prediction still works correctly.")
                        
                        # Show a simple explanation instead
                        with st.expander("❓ Why might visualization fail?"):
                            st.markdown("""
                            Feature visualization can fail for several reasons:
                            - **🏢 Model Architecture**: Some layers may not be compatible with GradCAM
                            - **🔄 PyTorch Version**: Newer versions may have different hook behaviors  
                            - **💾 Memory Constraints**: Visualization requires additional computation
                            - **🏷️ Layer Names**: The target layers might not exist in this model variant
                            
                            💪 **The core deepfake detection still works perfectly!**
                            """)
            
            # Risk assessment
            st.markdown("### ⚠️ Risk Assessment")
            if confidence > 0.9:
                risk_level = "Very High"
                risk_color = "#ff4757"
            elif confidence > 0.8:
                risk_level = "High"
                risk_color = "#ff6b6b"
            elif confidence > 0.7:
                risk_level = "Medium"
                risk_color = "#feca57"
            else:
                risk_level = "Low"
                risk_color = "#5f27cd"
            
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: {risk_color};">Confidence Level: {risk_level}</h4>
                    <p>Based on model certainty: {confidence*100:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Additional information section
    st.markdown("---")
    
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
        - **Input Resolution**: 224×224 pixels
        - **Output**: Binary classification (Real vs Fake) with confidence scores
        
        ### Limitations:
        - Performance may vary with image quality and resolution
        - New deepfake techniques may not be detected
        - Best results with clear, front-facing facial images
        - Multiple faces in one image may affect accuracy
        
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #666;">
            <p>🔬 Built with ❤️ by Aditi Salvi  | 🤖 Powered by EfficientNet-B3</p>
            <p>🛡️ Protecting digital authenticity, one image at a time</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
