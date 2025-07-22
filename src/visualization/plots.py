"""
Plotting utilities for the DeepFake Detective project.

This module contains functions for creating various plots and charts
used in the Streamlit application.
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, List, Any
import logging

from ..config.config import config

# Setup logging
logger = logging.getLogger(__name__)

class PlotGenerator:
    """
    Class for generating various plots and visualizations.
    """
    
    def __init__(self):
        """Initialize the plot generator with theme settings."""
        self.theme_colors = {
            'primary': config.app.primary_color,
            'secondary': config.app.secondary_color,
            'accent': config.app.accent_color,
            'success': config.app.success_color,
            'warning': config.app.warning_color,
            'danger': config.app.danger_color
        }
    
    def create_confidence_chart(self, real_prob: float, fake_prob: float) -> go.Figure:
        """
        Create a confidence chart showing real vs fake probabilities.
        
        Args:
            real_prob: Probability that the image is real
            fake_prob: Probability that the image is fake
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Add bar chart
        fig.add_trace(go.Bar(
            x=['Real', 'Fake'],
            y=[real_prob * 100, fake_prob * 100],
            marker_color=[self.theme_colors['success'], self.theme_colors['danger']],
            text=[f'{real_prob*100:.1f}%', f'{fake_prob*100:.1f}%'],
            textposition='auto',
            textfont=dict(size=14, color='white'),
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="🎯 Prediction Confidence",
                font=dict(size=20, family='Arial Black'),
                x=0.2
            ),
            xaxis=dict(
                title="Classification",
                title_font=dict(size=14),
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                title="Confidence (%)",
                title_font=dict(size=14),
                tickfont=dict(size=12),
                range=[0, 100]
            ),
            template="plotly_dark",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False
        )
        
        return fig
    
    def create_attention_plot(self, attention_maps: Dict[str, Dict[str, Any]]) -> go.Figure:
        """
        Create an interactive plot showing attention across different layers.
        
        Args:
            attention_maps: Dictionary containing attention data for each layer
            
        Returns:
            Plotly figure object
        """
        layer_names = list(attention_maps.keys())
        attention_scores = [np.mean(attention_maps[layer]['heatmap']) for layer in layer_names]
        
        # Create gradient colors
        colors = [
            self.theme_colors['primary'],
            self.theme_colors['secondary'], 
            self.theme_colors['accent'],
            self.theme_colors['warning']
        ][:len(layer_names)]
        
        fig = go.Figure()
        
        # Add bars with gradient effect
        fig.add_trace(go.Bar(
            x=[f"Layer {i+1}" for i in range(len(layer_names))],
            y=attention_scores,
            marker=dict(
                color=colors,
                line=dict(color='rgba(255,255,255,0.3)', width=1)
            ),
            text=[f'{score:.3f}' for score in attention_scores],
            textposition='auto',
            textfont=dict(size=12, color='white'),
            hovertemplate='<b>%{x}</b><br>Layer: %{customdata}<br>Avg Attention: %{y:.3f}<extra></extra>',
            customdata=layer_names
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="🧠 Model Attention Across Layers",
                font=dict(size=20, family='Arial Black'),
                x=0.1
            ),
            xaxis=dict(
                title="Network Layers",
                title_font=dict(size=14),
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                title="Average Attention Score",
                title_font=dict(size=14),
                tickfont=dict(size=12)
            ),
            template="plotly_dark",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False
        )
        
        return fig
    
    def create_risk_assessment_gauge(self, confidence: float) -> go.Figure:
        """
        Create a gauge chart for risk assessment.
        
        Args:
            confidence: Model confidence score (0-1)
            
        Returns:
            Plotly figure object
        """
        # Determine risk level and color
        if confidence > config.app.risk_thresholds["very_high"]:
            risk_level = "Very High"
            color = self.theme_colors['danger']
        elif confidence > config.app.risk_thresholds["high"]:
            risk_level = "High" 
            color = self.theme_colors['warning']
        elif confidence > config.app.risk_thresholds["medium"]:
            risk_level = "Medium"
            color = self.theme_colors['accent']
        else:
            risk_level = "Low"
            color = self.theme_colors['success']
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"🎯 Risk Level: {risk_level}"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return fig
    
    def create_layer_comparison_heatmap(self, attention_stats: Dict[str, Dict[str, float]]) -> go.Figure:
        """
        Create a heatmap comparing attention statistics across layers.
        
        Args:
            attention_stats: Dictionary of attention statistics for each layer
            
        Returns:
            Plotly figure object
        """
        layers = list(attention_stats.keys())
        metrics = ['mean_attention', 'max_attention', 'std_attention', 'attention_coverage']
        
        # Create matrix for heatmap
        z_matrix = []
        for metric in metrics:
            row = [attention_stats[layer][metric] for layer in layers]
            z_matrix.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_matrix,
            x=layers,
            y=['Mean', 'Max', 'Std', 'Coverage'],
            colorscale='Viridis',
            showscale=True,
            hoverongaps=False,
            hovertemplate='Layer: %{x}<br>Metric: %{y}<br>Value: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text="📊 Attention Statistics Heatmap",
                font=dict(size=18),
                x=0.5
            ),
            xaxis_title="Layers",
            yaxis_title="Attention Metrics",
            template="plotly_dark",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return fig
    
    def create_prediction_timeline(self, predictions: List[Dict[str, Any]]) -> go.Figure:
        """
        Create a timeline showing prediction history.
        
        Args:
            predictions: List of prediction dictionaries with timestamps
            
        Returns:
            Plotly figure object
        """
        if not predictions:
            # Return empty figure if no predictions
            fig = go.Figure()
            fig.add_annotation(
                text="No predictions yet",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16, color='white')
            )
            return fig
        
        timestamps = [pred['timestamp'] for pred in predictions]
        confidences = [pred['confidence'] for pred in predictions]
        predictions_class = [pred['predicted_class'] for pred in predictions]
        
        # Create colors based on prediction class
        colors = [self.theme_colors['success'] if pred == 0 else self.theme_colors['danger'] 
                 for pred in predictions_class]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=confidences,
            mode='markers+lines',
            marker=dict(
                size=10,
                color=colors,
                line=dict(color='white', width=1)
            ),
            line=dict(color='rgba(255,255,255,0.5)', width=2),
            text=[f"Class: {'Real' if pred == 0 else 'Fake'}" for pred in predictions_class],
            hovertemplate='Time: %{x}<br>Confidence: %{y:.1%}<br>%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title="📈 Prediction Timeline",
            xaxis_title="Time",
            yaxis_title="Confidence",
            template="plotly_dark",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return fig

class ChartThemes:
    """
    Predefined chart themes for consistent styling.
    """
    
    @staticmethod
    def get_dark_theme() -> dict:
        """Get dark theme configuration."""
        return {
            'template': 'plotly_dark',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': 'white'}
        }
    
    @staticmethod
    def get_light_theme() -> dict:
        """Get light theme configuration."""
        return {
            'template': 'plotly_white',
            'paper_bgcolor': 'rgba(255,255,255,0)',
            'plot_bgcolor': 'rgba(255,255,255,0)',
            'font': {'color': 'black'}
        }
