import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Optional
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Visualizer:
    """
    Handles visualization of data analysis and model results
    """
    
    def __init__(self):
        """Initialize the visualizer with default styling"""
        self.color_palette = {
            'primary': '#FF6B6B',
            'secondary': '#4ECDC4',
            'accent': '#45B7D1',
            'warning': '#F9E79F',
            'danger': '#EC7063',
            'success': '#58D68D'
        }
    
    def show_data_overview(self, data: pd.DataFrame) -> None:
        """
        Display comprehensive data overview visualizations
        
        Args:
            data (pd.DataFrame): Processed review data
        """
        st.subheader("Data Overview")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reviews", len(data))
        
        with col2:
            avg_length = data.get('text_length', pd.Series()).mean()
            st.metric("Avg Review Length", f"{avg_length:.0f}" if not pd.isna(avg_length) else "N/A")
        
        with col3:
            avg_words = data.get('word_count', pd.Series()).mean()
            st.metric("Avg Word Count", f"{avg_words:.0f}" if not pd.isna(avg_words) else "N/A")
        
        with col4:
            if 'rating' in data.columns:
                avg_rating = data['rating'].mean()
                st.metric("Avg Rating", f"{avg_rating:.1f}" if not pd.isna(avg_rating) else "N/A")
            else:
                st.metric("Data Quality", "Good")
        
        # Text length distribution
        if 'text_length' in data.columns:
            fig = px.histogram(
                data, 
                x='text_length',
                title='Review Length Distribution',
                labels={'text_length': 'Characters', 'count': 'Number of Reviews'},
                color_discrete_sequence=[self.color_palette['primary']]
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Rating distribution (if available)
        if 'rating' in data.columns:
            rating_counts = data['rating'].value_counts().sort_index()
            fig = px.bar(
                x=rating_counts.index,
                y=rating_counts.values,
                title='Rating Distribution',
                labels={'x': 'Rating', 'y': 'Number of Reviews'},
                color_discrete_sequence=[self.color_palette['secondary']]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlation heatmap
        self._show_feature_correlation(data)
    
    def _show_feature_correlation(self, data: pd.DataFrame) -> None:
        """Show correlation heatmap of numerical features"""
        # Select numerical features
        numerical_features = data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numerical_features if col not in ['review_id', 'rating']]
        
        if len(feature_cols) > 3:
            st.subheader("Feature Correlations")
            
            # Calculate correlation matrix
            corr_data = data[feature_cols].corr()
            
            # Create correlation heatmap using plotly
            fig = px.imshow(
                corr_data,
                text_auto=True,
                aspect="auto",
                title="Feature Correlation Matrix",
                color_continuous_scale="RdBu_r"
            )
            fig.update_layout(width=800, height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    def show_violation_distribution(self, predictions: pd.DataFrame) -> None:
        """
        Display distribution of policy violations
        
        Args:
            predictions (pd.DataFrame): Predictions with violation flags
        """
        st.subheader("Policy Violation Distribution")
        
        # Count violations
        violation_counts = {}
        violation_types = ['advertisement', 'irrelevant', 'rant']
        
        for violation_type in violation_types:
            col_name = f"{violation_type}_violation"
            if col_name in predictions.columns:
                violation_counts[violation_type.title()] = predictions[col_name].sum()
        
        if violation_counts:
            # Create violation distribution chart
            fig = px.bar(
                x=list(violation_counts.keys()),
                y=list(violation_counts.values()),
                title="Number of Violations by Type",
                labels={'x': 'Violation Type', 'y': 'Number of Violations'},
                color=list(violation_counts.values()),
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Violation rates
            total_reviews = len(predictions)
            violation_rates = {k: v/total_reviews*100 for k, v in violation_counts.items()}
            
            fig_rates = px.pie(
                values=list(violation_rates.values()),
                names=list(violation_rates.keys()),
                title="Violation Rates (%)"
            )
            st.plotly_chart(fig_rates, use_container_width=True)
    
    def show_confidence_distribution(self, predictions: pd.DataFrame) -> None:
        """Show distribution of confidence scores"""
        st.subheader("Confidence Score Distribution")
        
        confidence_cols = [col for col in predictions.columns if 'confidence' in col]
        
        if confidence_cols:
            fig = make_subplots(
                rows=1, 
                cols=len(confidence_cols),
                subplot_titles=[col.replace('_confidence', '').title() for col in confidence_cols]
            )
            
            for i, col in enumerate(confidence_cols):
                fig.add_trace(
                    go.Histogram(
                        x=predictions[col],
                        name=col.replace('_confidence', '').title(),
                        nbinsx=20
                    ),
                    row=1, col=i+1
                )
            
            fig.update_layout(
                title="Confidence Score Distributions",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def show_evaluation_metrics(self, metrics: Dict) -> None:
        """
        Display evaluation metrics in an organized format
        
        Args:
            metrics (Dict): Evaluation metrics from Evaluator
        """
        st.subheader("Model Performance Metrics")
        
        # Create metrics summary table
        metrics_data = []
        
        for violation_type, type_metrics in metrics.items():
            if violation_type == 'overall':
                continue
                
            if 'precision' in type_metrics and 'recall' in type_metrics:
                metrics_data.append({
                    'Violation Type': violation_type.title(),
                    'Accuracy': f"{type_metrics.get('accuracy', 0):.3f}",
                    'Precision': f"{type_metrics['precision'].get('macro', 0):.3f}",
                    'Recall': f"{type_metrics['recall'].get('macro', 0):.3f}",
                    'F1-Score': f"{type_metrics['f1_score'].get('macro', 0):.3f}",
                    'Samples': type_metrics.get('support', {}).get('total', 0)
                })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Visualize metrics
            fig = go.Figure()
            
            violation_types = metrics_df['Violation Type'].tolist()
            
            fig.add_trace(go.Bar(
                name='Precision',
                x=violation_types,
                y=[float(x) for x in metrics_df['Precision'].tolist()],
                marker_color=self.color_palette['primary']
            ))
            
            fig.add_trace(go.Bar(
                name='Recall',
                x=violation_types,
                y=[float(x) for x in metrics_df['Recall'].tolist()],
                marker_color=self.color_palette['secondary']
            ))
            
            fig.add_trace(go.Bar(
                name='F1-Score',
                x=violation_types,
                y=[float(x) for x in metrics_df['F1-Score'].tolist()],
                marker_color=self.color_palette['accent']
            ))
            
            fig.update_layout(
                title='Model Performance by Violation Type',
                xaxis_title='Violation Type',
                yaxis_title='Score',
                barmode='group',
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def show_confusion_matrices(self, predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> None:
        """
        Display confusion matrices for each violation type
        
        Args:
            predictions (pd.DataFrame): Model predictions
            ground_truth (pd.DataFrame): Ground truth labels
        """
        st.subheader("Confusion Matrices")
        
        violation_types = ['advertisement', 'irrelevant', 'rant']
        cols = st.columns(len(violation_types))
        
        for i, violation_type in enumerate(violation_types):
            pred_col = f"{violation_type}_violation"
            gt_col = f"{violation_type}_gt"
            
            if pred_col in predictions.columns and gt_col in ground_truth.columns:
                with cols[i]:
                    # Create confusion matrix
                    from sklearn.metrics import confusion_matrix
                    
                    y_true = ground_truth[gt_col]
                    y_pred = predictions[pred_col]
                    
                    cm = confusion_matrix(y_true, y_pred)
                    
                    # Create heatmap
                    fig = px.imshow(
                        cm,
                        text_auto=True,
                        aspect="auto",
                        title=f"{violation_type.title()} Confusion Matrix",
                        labels=dict(x="Predicted", y="Actual"),
                        x=['No Violation', 'Violation'],
                        y=['No Violation', 'Violation'],
                        color_continuous_scale="Blues"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    def show_detailed_analysis(self, predictions: pd.DataFrame) -> None:
        """
        Show detailed analysis of predictions and patterns
        
        Args:
            predictions (pd.DataFrame): Prediction results
        """
        st.subheader("Detailed Analysis")
        
        # Confidence vs violations
        self.show_confidence_distribution(predictions)
        
        # Feature importance for violations
        self._show_feature_violation_relationship(predictions)
        
        # Review length vs violations
        self._show_length_violation_analysis(predictions)
    
    def _show_feature_violation_relationship(self, predictions: pd.DataFrame) -> None:
        """Show relationship between features and violations"""
        st.subheader("Feature-Violation Relationships")
        
        feature_cols = ['promotional_words', 'has_url', 'first_person_count', 'visited_indicators']
        violation_cols = [col for col in predictions.columns if 'violation' in col and col.endswith('_violation')]
        
        # Filter available columns
        available_features = [col for col in feature_cols if col in predictions.columns]
        
        if available_features and violation_cols:
            # Create subplot for each violation type
            n_violations = len(violation_cols)
            fig = make_subplots(
                rows=1, 
                cols=n_violations,
                subplot_titles=[col.replace('_violation', '').title() for col in violation_cols]
            )
            
            for i, violation_col in enumerate(violation_cols):
                # Calculate mean feature values for violated vs non-violated reviews
                violated = predictions[predictions[violation_col] == True]
                non_violated = predictions[predictions[violation_col] == False]
                
                for feature in available_features[:2]:  # Limit to 2 features for clarity
                    if feature in predictions.columns:
                        violated_mean = violated[feature].mean()
                        non_violated_mean = non_violated[feature].mean()
                        
                        fig.add_trace(
                            go.Bar(
                                name=f"{feature} (Violated)",
                                x=[feature],
                                y=[violated_mean],
                                marker_color=self.color_palette['danger']
                            ),
                            row=1, col=i+1
                        )
                        
                        fig.add_trace(
                            go.Bar(
                                name=f"{feature} (Non-violated)",
                                x=[feature],
                                y=[non_violated_mean],
                                marker_color=self.color_palette['success']
                            ),
                            row=1, col=i+1
                        )
            
            fig.update_layout(
                title="Feature Distributions: Violated vs Non-Violated Reviews",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_length_violation_analysis(self, predictions: pd.DataFrame) -> None:
        """Analyze relationship between review length and violations"""
        if 'text_length' not in predictions.columns:
            return
            
        st.subheader("Review Length vs Violations")
        
        violation_cols = [col for col in predictions.columns if 'violation' in col and col.endswith('_violation')]
        
        if violation_cols:
            length_data = []
            
            for violation_col in violation_cols:
                violation_type = violation_col.replace('_violation', '')
                
                violated_lengths = predictions[predictions[violation_col] == True]['text_length']
                non_violated_lengths = predictions[predictions[violation_col] == False]['text_length']
                
                # Add data for violated reviews
                for length in violated_lengths:
                    length_data.append({
                        'Violation Type': violation_type,
                        'Length': length,
                        'Violated': 'Yes'
                    })
                
                # Sample non-violated for balance
                sample_size = min(len(violated_lengths) * 2, len(non_violated_lengths))
                sampled_non_violated = non_violated_lengths.sample(n=sample_size) if sample_size > 0 else non_violated_lengths
                
                for length in sampled_non_violated:
                    length_data.append({
                        'Violation Type': violation_type,
                        'Length': length,
                        'Violated': 'No'
                    })
            
            if length_data:
                length_df = pd.DataFrame(length_data)
                
                fig = px.box(
                    length_df,
                    x='Violation Type',
                    y='Length',
                    color='Violated',
                    title="Review Length Distribution by Violation Type"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def show_model_insights(self, predictions: pd.DataFrame, original_data: pd.DataFrame) -> None:
        """
        Show insights about model behavior and decision patterns
        
        Args:
            predictions (pd.DataFrame): Model predictions
            original_data (pd.DataFrame): Original processed data
        """
        st.subheader("Model Insights")
        
        # High confidence predictions analysis
        st.write("**High Confidence Predictions:**")
        
        confidence_threshold = 0.8
        high_conf_ads = predictions[predictions.get('advertisement_confidence', 0) > confidence_threshold]
        high_conf_irrelevant = predictions[predictions.get('irrelevant_confidence', 0) > confidence_threshold]
        high_conf_rants = predictions[predictions.get('rant_confidence', 0) > confidence_threshold]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "High Conf. Advertisements", 
                len(high_conf_ads),
                f"{len(high_conf_ads)/len(predictions)*100:.1f}%"
            )
        
        with col2:
            st.metric(
                "High Conf. Irrelevant", 
                len(high_conf_irrelevant),
                f"{len(high_conf_irrelevant)/len(predictions)*100:.1f}%"
            )
        
        with col3:
            st.metric(
                "High Conf. Rants", 
                len(high_conf_rants),
                f"{len(high_conf_rants)/len(predictions)*100:.1f}%"
            )
        
        # Show sample high-confidence predictions
        if len(high_conf_ads) > 0:
            st.write("**Sample High-Confidence Advertisement Violations:**")
            sample_ads = high_conf_ads.head(3)[['review_text', 'advertisement_confidence', 'advertisement_reason']]
            st.dataframe(sample_ads, use_container_width=True)
        
        # Decision boundary analysis
        self._show_decision_boundary_analysis(predictions)
    
    def _show_decision_boundary_analysis(self, predictions: pd.DataFrame) -> None:
        """Analyze model decision boundaries"""
        st.write("**Decision Boundary Analysis:**")
        
        # Confidence score distributions
        confidence_cols = [col for col in predictions.columns if 'confidence' in col]
        
        if confidence_cols:
            threshold = 0.5
            
            boundary_data = []
            for col in confidence_cols:
                violation_type = col.replace('_confidence', '')
                confidences = predictions[col]
                
                near_boundary = confidences[(confidences > threshold - 0.1) & (confidences < threshold + 0.1)]
                high_confidence = confidences[confidences > 0.8]
                low_confidence = confidences[confidences < 0.2]
                
                boundary_data.append({
                    'Violation Type': violation_type.title(),
                    'Near Boundary (±0.1)': len(near_boundary),
                    'High Confidence (>0.8)': len(high_confidence),
                    'Low Confidence (<0.2)': len(low_confidence)
                })
            
            boundary_df = pd.DataFrame(boundary_data)
            st.dataframe(boundary_df, use_container_width=True)
