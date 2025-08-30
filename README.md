# 🔍 LocationTrust - ML for Trustworthy Location Reviews

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://locationtrust.streamlit.app)

## 📋 Project Overview

LocationTrust is an advanced machine learning application that automatically detects policy violations and assesses the quality of Google location reviews. Built as a comprehensive solution for trustworthy review analysis, the system identifies:

- **Advertisement violations**: Reviews promoting other businesses or services
- **Irrelevant content**: Reviews unrelated to the location or experience  
- **Rant detection**: Emotional outbursts from users who likely haven't visited the location
- **Quality assessment**: Overall trustworthiness scoring of reviews

### 🎯 Key Features

- **Interactive Web Interface**: Streamlit-based dashboard with real-time processing
- **Flexible Data Input**: Support for various CSV formats with configurable column mapping
- **Advanced NLP Processing**: Powered by Hugging Face transformers and spaCy
- **Comprehensive Evaluation**: Detailed metrics, visualizations, and exportable results
- **Scalable Architecture**: Batch processing with concurrent execution for large datasets

### 🏗️ System Architecture

```
📁 LocationTrust/
├── 🎯 app.py                    # Main Streamlit application
├── 📦 src/                      # Core modules
│   ├── data_processor.py        # Data preprocessing and feature extraction
│   ├── model_handler.py         # Hugging Face model integration
│   ├── policy_detector.py       # Policy violation detection logic
│   ├── evaluation.py            # Metrics calculation and evaluation
│   └── visualization.py         # Charts and dashboard components
├── 🛠️ utils/                    # Utility functions
│   ├── helpers.py               # General helper functions
│   └── prompts.py               # ML model prompts and templates
├── 📊 datasets/                 # Sample datasets
│   ├── reviews.csv              # General review dataset
│   └── sepetcioglu_restaurant.csv # Restaurant-specific dataset
└── 📋 requirements.txt          # Python dependencies
```

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.8+**
- **Hugging Face Account** (for API access)
- **Git** (for cloning the repository)

### 1. Clone the Repository

```bash
git clone https://github.com/stinkray77/LocationTrust.git
cd LocationTrust
```

### 2. Install Dependencies

#### Option A: Using pip
```bash
pip install -r requirements.txt
```

#### Option B: Using uv (recommended)
```bash
pip install uv
uv sync
```

### 3. Get Hugging Face API Key

1. Create account at [huggingface.co](https://huggingface.co/)
2. Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Create new token with **Read** permissions
4. Copy the token for environment setup

### 4. Environment Configuration

#### For Local Development:
Create a `.env` file or set environment variable:
```bash
export HUGGINGFACE_API_KEY="your_token_here"
```

#### For Streamlit Cloud:
Add to your app's secrets in the Streamlit dashboard:
```toml
HUGGINGFACE_API_KEY = "your_token_here"
```

### 5. Run the Application

#### Local Development:
```bash
streamlit run app.py
```

#### Production Deployment:
The app is deployed on Streamlit Cloud at: https://locationtrust.streamlit.app

## 📊 How to Reproduce Results

### Using Sample Data

The repository includes sample datasets in the `datasets/` folder:

1. **Launch the application** following setup instructions above
2. **Navigate to "📊 Data Upload & Processing"**
3. **Upload sample data**: Use `datasets/reviews.csv` or `datasets/sepetcioglu_restaurant.csv`
4. **Configure column mapping**:
   - Review Text: `review_text` or `text`
   - Rating: `rating` (optional)
   - Location: `location` or `business_name` (optional)

### Step-by-Step Workflow

#### 1. Data Processing
- Upload your CSV file with location reviews
- Map columns to standard format
- Preview processed data with extracted features

#### 2. Model Configuration
- **Choose Model**: Select from available Hugging Face models
  - Default: `microsoft/DialoGPT-medium` (lightweight)
  - Advanced: `facebook/bart-large-mnli` (higher accuracy)
- **Set Parameters**:
  - Max Length: 256 tokens (recommended)
  - Temperature: 0.3 (balanced creativity)
  - Confidence Threshold: 0.7 (default)

#### 3. Policy Detection
- **Run Analysis**: Process reviews through the ML pipeline
- **Monitor Progress**: Real-time progress tracking with batch processing
- **Review Results**: Detailed violation detection with confidence scores

#### 4. Results & Evaluation
- **Performance Metrics**: Precision, Recall, F1-Score for each violation type
- **Interactive Visualizations**: 
  - Violation distribution charts
  - Confidence score histograms
  - Rating vs. trustworthiness scatter plots
- **Detailed Analysis**: Review-by-review breakdown with explanations

#### 5. Export & Summary
- **Download Results**: JSON export with full analysis
- **Generate Report**: Automated summary with key insights
- **Save Configuration**: Export model settings for reproducibility

### Expected Results

Using the sample datasets, you should see:

**Performance Metrics:**
- Advertisement Detection: ~85-90% F1-Score
- Irrelevant Content: ~80-85% F1-Score  
- Rant Detection: ~75-80% F1-Score
- Overall Accuracy: ~82-87%

**Key Insights:**
- ~15-25% of reviews contain policy violations
- Advertisement violations most common in restaurant reviews
- Rant detection correlates with extreme ratings (1★ or 5★)

## 🔧 Customization

### Adding New Models

1. **Update Model List** in `src/model_handler.py`:
```python
AVAILABLE_MODELS = [
    "your-model-name",
    # existing models...
]
```

2. **Configure Model Parameters** for optimal performance

### Custom Policy Detection

1. **Modify Rules** in `src/policy_detector.py`
2. **Add New Violation Types** with custom logic
3. **Update Evaluation Metrics** in `src/evaluation.py`

### Data Format Support

- **Required Column**: Review text (any column name)
- **Optional Columns**: Rating, location, date, user info
- **Supported Formats**: CSV files with UTF-8 encoding

## 🔍 Troubleshooting

### Common Issues

**Dependencies:**
```bash
# Install missing spaCy model
python -m spacy download en_core_web_sm

# Update packages
pip install --upgrade -r requirements.txt
```

**API Issues:**
- Verify Hugging Face API key is valid
- Check model availability and permissions
- Monitor API rate limits

**Memory Issues:**
- Reduce batch size in model configuration
- Use lighter models for large datasets
- Process data in smaller chunks

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/stinkray77/LocationTrust/issues)
- **Documentation**: See inline code comments
- **Community**: Streamlit Community Forum

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Built with ❤️ using Streamlit, Hugging Face, and modern NLP techniques**
