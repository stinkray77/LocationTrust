import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import json
from datetime import datetime
import os

from src.data_processor import DataProcessor
from src.model_handler import ModelHandler
from src.policy_detector import PolicyDetector
from src.evaluation import Evaluator
from src.visualization import Visualizer

# Page config
st.set_page_config(
    page_title="TrustIssues - ML for Trustworthy Location Reviews",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'model_handler' not in st.session_state:
    st.session_state.model_handler = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None

def main():
    # Header
    st.title("🔍 TrustIssues")
    st.markdown("### ML for Trustworthy Location Reviews")
    st.markdown("Detect policy violations and assess quality in Google location reviews using advanced NLP models.")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section",
        ["📊 Data Upload & Processing", "🤖 Model Configuration", "🔍 Policy Detection", "📈 Results & Evaluation", "📋 Export & Summary"]
    )
    
    if page == "📊 Data Upload & Processing":
        data_upload_page()
    elif page == "🤖 Model Configuration":
        model_configuration_page()
    elif page == "🔍 Policy Detection":
        policy_detection_page()
    elif page == "📈 Results & Evaluation":
        results_evaluation_page()
    elif page == "📋 Export & Summary":
        export_summary_page()

def data_upload_page():
    st.header("📊 Data Upload & Processing")
    
    # File upload
    st.subheader("Upload Review Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV file containing location reviews",
        type=['csv'],
        help="Upload Google location reviews or similar datasets. Expected columns: review_text, rating, location, etc."
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            
            st.success(f"Successfully loaded {len(data)} reviews!")
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Reviews", len(data))
            with col2:
                st.metric("Columns", len(data.columns))
            with col3:
                if 'rating' in data.columns:
                    st.metric("Avg Rating", f"{data['rating'].mean():.1f}")
                else:
                    st.metric("Unique Values", len(data.iloc[:, 0].unique()))
            
            # Column mapping
            st.subheader("Column Mapping")
            col1, col2 = st.columns(2)
            
            with col1:
                review_column = st.selectbox(
                    "Review Text Column",
                    options=data.columns,
                    help="Select the column containing review text"
                )
            
            with col2:
                rating_column = st.selectbox(
                    "Rating Column (optional)",
                    options=['None'] + list(data.columns),
                    help="Select the column containing ratings"
                )
            
            # Additional metadata columns
            st.subheader("Metadata Columns (Optional)")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                location_column = st.selectbox(
                    "Location/Business Column",
                    options=['None'] + list(data.columns),
                    help="Column containing business/location names"
                )
            
            with col2:
                timestamp_column = st.selectbox(
                    "Timestamp Column",
                    options=['None'] + list(data.columns),
                    help="Column containing review timestamps"
                )
            
            with col3:
                user_column = st.selectbox(
                    "User ID Column",
                    options=['None'] + list(data.columns),
                    help="Column containing user identifiers"
                )
            
            # Process data
            if st.button("Process Data", type="primary"):
                with st.spinner("Processing data..."):
                    processor = DataProcessor()
                    
                    # Create column mapping
                    column_mapping = {
                        'review_text': review_column,
                        'rating': rating_column if rating_column != 'None' else None,
                        'location': location_column if location_column != 'None' else None,
                        'timestamp': timestamp_column if timestamp_column != 'None' else None,
                        'user_id': user_column if user_column != 'None' else None
                    }
                    
                    processed_data = processor.process_data(data, column_mapping)
                    st.session_state.processed_data = processed_data
                    
                    st.success("Data processed successfully!")
                    st.rerun()
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    # Display processed data
    if st.session_state.processed_data is not None:
        st.subheader("Processed Data Preview")
        st.dataframe(st.session_state.processed_data.head(), use_container_width=True)
        
        # Data statistics
        st.subheader("Data Statistics")
        viz = Visualizer()
        viz.show_data_overview(st.session_state.processed_data)

def model_configuration_page():
    st.header("🤖 Model Configuration")
    
    if st.session_state.processed_data is None:
        st.warning("Please upload and process data first!")
        return
    
    st.subheader("Model Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_choice = st.selectbox(
            "Choose Model",
            ["google/gemma-2-2b-it", "Qwen/Qwen2-7B-Instruct", "microsoft/DialoGPT-medium"],
            help="Select the Hugging Face model for classification"
        )
    
    with col2:
        use_api_key = st.checkbox(
            "Use Hugging Face API",
            value=True,
            help="Use Hugging Face Inference API (requires API key)"
        )
    
    # API Key input
    if use_api_key:
        hf_api_key = st.text_input(
            "Hugging Face API Key",
            type="password",
            value=os.getenv("HUGGINGFACE_API_KEY", ""),
            help="Enter your Hugging Face API key for model inference"
        )
    else:
        hf_api_key = None
    
    # Model parameters
    st.subheader("Model Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_length = st.slider("Max Token Length", 128, 512, 256)
    
    with col2:
        temperature = st.slider("Temperature", 0.1, 1.0, 0.3, 0.1)
    
    with col3:
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.9, 0.7, 0.05)
    
    # Initialize model
    if st.button("Initialize Model", type="primary"):
        if use_api_key and not hf_api_key:
            st.error("Please provide a Hugging Face API key!")
            return
        
        with st.spinner("Initializing model..."):
            try:
                model_handler = ModelHandler(
                    model_name=model_choice,
                    api_key=hf_api_key,
                    max_length=max_length,
                    temperature=temperature,
                    confidence_threshold=confidence_threshold
                )
                
                st.session_state.model_handler = model_handler
                st.success(f"Model {model_choice} initialized successfully!")
                
                # Test model with sample text
                st.subheader("Model Test")
                test_text = st.text_area(
                    "Test the model with sample review:",
                    "Great pizza place! The food was delicious and service was excellent."
                )
                
                if st.button("Test Model"):
                    with st.spinner("Testing model..."):
                        test_result = model_handler.classify_review(test_text)
                        st.json(test_result)
                        
            except Exception as e:
                st.error(f"Error initializing model: {str(e)}")

def policy_detection_page():
    st.header("🔍 Policy Detection")
    
    if st.session_state.processed_data is None:
        st.warning("Please upload and process data first!")
        return
    
    if st.session_state.model_handler is None:
        st.warning("Please initialize the model first!")
        return
    
    # Policy configuration
    st.subheader("Policy Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Detection Policies:**")
        detect_ads = st.checkbox("🚫 Advertisement Detection", value=True)
        detect_irrelevant = st.checkbox("❓ Irrelevant Content Detection", value=True)
        detect_rants = st.checkbox("😤 Rant Without Visit Detection", value=True)
    
    with col2:
        st.markdown("**Processing Options:**")
        batch_size = st.number_input("Batch Size", min_value=1, max_value=100, value=10)
        show_progress = st.checkbox("Show Progress", value=True)
        save_intermediate = st.checkbox("Save Intermediate Results", value=False)
    
    # Run detection
    if st.button("Run Policy Detection", type="primary"):
        with st.spinner("Running policy detection..."):
            try:
                policy_detector = PolicyDetector(st.session_state.model_handler)
                
                # Configure policies
                policies = {
                    'advertisement': detect_ads,
                    'irrelevant': detect_irrelevant,
                    'rant_without_visit': detect_rants
                }
                
                # Run detection
                predictions = policy_detector.detect_violations(
                    st.session_state.processed_data,
                    policies=policies,
                    batch_size=batch_size,
                    show_progress=show_progress
                )
                
                st.session_state.predictions = predictions
                st.success(f"Policy detection completed! Processed {len(predictions)} reviews.")
                
                # Show summary
                st.subheader("Detection Summary")
                summary_df = policy_detector.get_summary_stats(predictions)
                st.dataframe(summary_df, use_container_width=True)
                
                # Visualize results
                viz = Visualizer()
                viz.show_violation_distribution(predictions)
                
            except Exception as e:
                st.error(f"Error during policy detection: {str(e)}")
    
    # Show sample predictions
    if st.session_state.predictions is not None:
        st.subheader("Sample Predictions")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            violation_filter = st.selectbox(
                "Filter by violation type:",
                ["All", "Advertisement", "Irrelevant", "Rant Without Visit", "No Violations"]
            )
        
        with col2:
            confidence_filter = st.slider("Minimum confidence:", 0.0, 1.0, 0.5, 0.1)
        
        # Apply filters
        filtered_predictions = st.session_state.predictions.copy()
        
        if violation_filter != "All":
            if violation_filter == "No Violations":
                filtered_predictions = filtered_predictions[
                    (filtered_predictions['advertisement_violation'] == False) &
                    (filtered_predictions['irrelevant_violation'] == False) &
                    (filtered_predictions['rant_violation'] == False)
                ]
            else:
                violation_col = f"{violation_filter.lower().replace(' ', '_')}_violation"
                filtered_predictions = filtered_predictions[filtered_predictions[violation_col] == True]
        
        # Apply confidence filter
        conf_cols = [col for col in filtered_predictions.columns if 'confidence' in col]
        for col in conf_cols:
            filtered_predictions = filtered_predictions[filtered_predictions[col] >= confidence_filter]
        
        st.dataframe(filtered_predictions.head(20), use_container_width=True)

def results_evaluation_page():
    st.header("📈 Results & Evaluation")
    
    if st.session_state.predictions is None:
        st.warning("Please run policy detection first!")
        return
    
    # Evaluation options
    st.subheader("Evaluation Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        has_ground_truth = st.checkbox(
            "I have ground truth labels",
            help="Check if you have manually labeled data for evaluation"
        )
    
    with col2:
        generate_pseudo_labels = st.checkbox(
            "Generate pseudo labels with advanced model",
            help="Use a more advanced model to generate labels for evaluation"
        )
    
    evaluator = Evaluator()
    viz = Visualizer()
    
    if has_ground_truth:
        st.subheader("Ground Truth Upload")
        gt_file = st.file_uploader(
            "Upload ground truth labels (CSV)",
            type=['csv'],
            help="CSV with columns: review_id, advertisement, irrelevant, rant_without_visit"
        )
        
        if gt_file is not None:
            try:
                ground_truth = pd.read_csv(gt_file)
                
                # Calculate metrics
                with st.spinner("Calculating evaluation metrics..."):
                    metrics = evaluator.calculate_metrics(st.session_state.predictions, ground_truth)
                    st.session_state.evaluation_results = metrics
                
                # Display metrics
                st.subheader("Performance Metrics")
                viz.show_evaluation_metrics(metrics)
                
                # Confusion matrices
                st.subheader("Confusion Matrices")
                viz.show_confusion_matrices(st.session_state.predictions, ground_truth)
                
            except Exception as e:
                st.error(f"Error processing ground truth: {str(e)}")
    
    elif generate_pseudo_labels:
        st.subheader("Pseudo Label Generation")
        st.info("This will use GPT-4 or similar model to generate pseudo labels for a subset of reviews.")
        
        sample_size = st.slider("Sample size for pseudo labeling", 50, 500, 100)
        
        if st.button("Generate Pseudo Labels"):
            with st.spinner("Generating pseudo labels..."):
                try:
                    # Sample data for pseudo labeling
                    sample_data = st.session_state.predictions.sample(n=sample_size)
                    
                    # This would typically call a more advanced model
                    pseudo_labels = evaluator.generate_pseudo_labels(sample_data)
                    
                    # Calculate metrics
                    metrics = evaluator.calculate_metrics(sample_data, pseudo_labels)
                    st.session_state.evaluation_results = metrics
                    
                    # Display results
                    st.subheader("Pseudo Label Evaluation")
                    viz.show_evaluation_metrics(metrics)
                    
                except Exception as e:
                    st.error(f"Error generating pseudo labels: {str(e)}")
    
    # Show distribution analysis
    st.subheader("Result Analysis")
    viz.show_detailed_analysis(st.session_state.predictions)
    
    # Model interpretation
    st.subheader("Model Insights")
    if st.session_state.model_handler:
        viz.show_model_insights(st.session_state.predictions, st.session_state.processed_data)

def export_summary_page():
    st.header("📋 Export & Summary")
    
    if st.session_state.predictions is None:
        st.warning("No predictions to export. Please run policy detection first!")
        return
    
    # Summary statistics
    st.subheader("Final Summary")
    
    total_reviews = len(st.session_state.predictions)
    violations = {
        'Advertisement': st.session_state.predictions['advertisement_violation'].sum(),
        'Irrelevant Content': st.session_state.predictions['irrelevant_violation'].sum(),
        'Rant Without Visit': st.session_state.predictions['rant_violation'].sum()
    }
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", total_reviews)
    
    with col2:
        st.metric("Advertisement Violations", violations['Advertisement'])
    
    with col3:
        st.metric("Irrelevant Content", violations['Irrelevant Content'])
    
    with col4:
        st.metric("Rant Violations", violations['Rant Without Visit'])
    
    # Violation rate chart
    violation_rates = {k: v/total_reviews*100 for k, v in violations.items()}
    fig = px.bar(
        x=list(violation_rates.keys()),
        y=list(violation_rates.values()),
        title="Policy Violation Rates (%)",
        labels={'x': 'Violation Type', 'y': 'Percentage (%)'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    st.subheader("Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export predictions
        if st.button("Download Predictions CSV", type="primary"):
            csv = st.session_state.predictions.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"trustissues_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        # Export summary report
        if st.button("Download Summary Report"):
            report = generate_summary_report()
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"trustissues_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Model information
    st.subheader("Model Information")
    if st.session_state.model_handler:
        model_info = {
            "Model": st.session_state.model_handler.model_name,
            "Total Reviews Processed": total_reviews,
            "Processing Time": "N/A",  # Could track this
            "Average Confidence": f"{st.session_state.predictions[['advertisement_confidence', 'irrelevant_confidence', 'rant_confidence']].mean().mean():.3f}"
        }
        
        for key, value in model_info.items():
            st.write(f"**{key}:** {value}")
    
    # Recommendations
    st.subheader("Recommendations")
    recommendations = generate_recommendations(violations, total_reviews)
    for rec in recommendations:
        st.write(f"• {rec}")
    
    # Hackathon submission checklist
    st.subheader("Hackathon Submission Checklist")
    st.markdown("""
    - ✅ **Text Description**: Ready for Devpost submission
    - ✅ **GitHub Repository**: Complete with documented code
    - ✅ **Demonstration Video**: System functionality shown
    - ✅ **Interactive Demo**: TrustIssues Streamlit app
    - ✅ **Evaluation Metrics**: Precision, recall, F1-score calculated
    - ✅ **Policy Enforcement**: Advertisement, irrelevant content, and rant detection
    - ✅ **Feature Engineering**: Text and metadata processing
    - ✅ **Hugging Face Integration**: Transformer models implemented
    """)

def generate_summary_report():
    """Generate a comprehensive summary report"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "name": st.session_state.model_handler.model_name if st.session_state.model_handler else "Unknown",
            "total_reviews": len(st.session_state.predictions),
        },
        "violation_summary": {
            "advertisement": int(st.session_state.predictions['advertisement_violation'].sum()),
            "irrelevant": int(st.session_state.predictions['irrelevant_violation'].sum()),
            "rant_without_visit": int(st.session_state.predictions['rant_violation'].sum()),
        },
        "confidence_stats": {
            "avg_advertisement_confidence": float(st.session_state.predictions['advertisement_confidence'].mean()),
            "avg_irrelevant_confidence": float(st.session_state.predictions['irrelevant_confidence'].mean()),
            "avg_rant_confidence": float(st.session_state.predictions['rant_confidence'].mean()),
        }
    }
    
    if st.session_state.evaluation_results:
        report["evaluation_metrics"] = st.session_state.evaluation_results
    
    return json.dumps(report, indent=2)

def generate_recommendations(violations, total_reviews):
    """Generate actionable recommendations based on results"""
    recommendations = []
    
    violation_rate = sum(violations.values()) / total_reviews
    
    if violation_rate > 0.3:
        recommendations.append("High violation rate detected. Consider implementing automated filtering.")
    
    if violations['Advertisement'] > total_reviews * 0.1:
        recommendations.append("Significant advertisement violations. Enhance promotional content detection.")
    
    if violations['Irrelevant Content'] > total_reviews * 0.15:
        recommendations.append("Many irrelevant reviews detected. Consider topic modeling for better relevance scoring.")
    
    if violations['Rant Without Visit'] > total_reviews * 0.05:
        recommendations.append("Rant detection active. Consider implementing user verification systems.")
    
    if violation_rate < 0.05:
        recommendations.append("Low violation rate indicates high-quality review dataset.")
    
    recommendations.append("Regular model retraining recommended to maintain detection accuracy.")
    recommendations.append("Consider implementing user feedback loop for continuous improvement.")
    
    return recommendations

if __name__ == "__main__":
    main()
