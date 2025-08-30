# TrustIssues - ML for Trustworthy Location Reviews

## Overview

TrustIssues is a Streamlit-based web application that uses Machine Learning and Natural Language Processing to automatically assess the quality and relevancy of Google location reviews. The system detects policy violations including advertisements, irrelevant content, and rants from users who likely haven't visited the reviewed location. Built for a ByteDance hackathon challenge, the application provides an end-to-end pipeline for data processing, model inference, policy detection, evaluation, and visualization of results.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for web interface
- **Layout**: Multi-page application with sidebar navigation
- **Session State Management**: Persistent storage of data, models, and results across user interactions
- **Interactive Components**: File upload, model configuration, real-time progress tracking, and dynamic visualizations

### Backend Architecture
- **Modular Design**: Separated into distinct modules (data_processor, model_handler, policy_detector, evaluation, visualization)
- **Data Processing Pipeline**: Sequential processing from raw data upload through feature extraction to final results
- **Model Integration**: Hugging Face API integration with fallback to local processing
- **Concurrent Processing**: ThreadPoolExecutor for batch processing of reviews
- **Error Handling**: Comprehensive logging and graceful error recovery

### NLP and ML Components
- **Text Processing**: spaCy integration for advanced NLP with fallback to basic text processing
- **Feature Engineering**: Automated extraction of textual and metadata features
- **Model Handler**: Flexible wrapper for Hugging Face transformers with configurable parameters
- **Policy Detection**: Rule-based and ML-based violation detection with confidence scoring
- **Evaluation Framework**: Comprehensive metrics calculation including precision, recall, F1-score

### Data Architecture
- **Input Flexibility**: Support for various CSV formats with configurable column mapping
- **Data Validation**: Input validation and error handling for malformed data
- **Feature Storage**: Extracted features stored alongside original data
- **Results Integration**: Predictions merged with original data for comprehensive analysis

### Visualization System
- **Interactive Charts**: Plotly-based visualizations for data exploration and results analysis
- **Dashboard Layout**: Multi-column layouts with metrics cards and dynamic charts
- **Export Capabilities**: JSON export functionality for results and configurations

## External Dependencies

### Core ML/NLP Libraries
- **Hugging Face**: Primary inference platform for transformer models via InferenceClient
- **spaCy**: Advanced NLP processing (en_core_web_sm model)
- **scikit-learn**: Evaluation metrics and model assessment

### Data Processing
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations

### Web Framework
- **Streamlit**: Primary web application framework
- **plotly**: Interactive data visualizations
- **seaborn/matplotlib**: Additional visualization support

### Utility Libraries
- **requests**: HTTP client for API communications
- **concurrent.futures**: Parallel processing capabilities
- **pathlib**: File system operations

### API Integrations
- **Hugging Face API**: Model inference with optional API key authentication
- **Model Support**: Designed to work with various transformer models for classification tasks

### Environment Requirements
- **Python Environment**: Requires Python with package management
- **Optional Models**: spaCy English model (graceful degradation if unavailable)
- **API Keys**: Optional Hugging Face API key for enhanced model access