"""
Utility functions and helpers for TrustIssues application
"""

import pandas as pd
import numpy as np
import re
import json
import logging
import os
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import hashlib
import unicodedata
from pathlib import Path
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextProcessor:
    """
    Text processing utilities for review analysis
    """
    
    @staticmethod
    def clean_text(text: str, remove_special_chars: bool = True) -> str:
        """
        Clean and normalize text for processing
        
        Args:
            text (str): Raw text to clean
            remove_special_chars (bool): Whether to remove special characters
            
        Returns:
            str: Cleaned text
        """
        if not text or pd.isna(text):
            return ""
        
        # Convert to string and normalize unicode
        text = str(text)
        text = unicodedata.normalize('NFKD', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters if requested
        if remove_special_chars:
            text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """Extract URLs from text"""
        if not text:
            return []
        
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            r'|www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            r'|(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s]*)?'
        )
        
        urls = url_pattern.findall(text)
        return [url.strip() for url in urls if url.strip()]
    
    @staticmethod
    def extract_phone_numbers(text: str) -> List[str]:
        """Extract phone numbers from text"""
        if not text:
            return []
        
        phone_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # XXX-XXX-XXXX or XXX.XXX.XXXX
            r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',    # (XXX) XXX-XXXX
            r'\b\d{3}\s\d{3}\s\d{4}\b',        # XXX XXX XXXX
            r'\+\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',  # International
        ]
        
        phones = []
        for pattern in phone_patterns:
            phones.extend(re.findall(pattern, text))
        
        return [phone.strip() for phone in phones if phone.strip()]
    
    @staticmethod
    def extract_emails(text: str) -> List[str]:
        """Extract email addresses from text"""
        if not text:
            return []
        
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text, re.IGNORECASE)
        
        return [email.strip().lower() for email in emails if email.strip()]
    
    @staticmethod
    def count_sentences(text: str) -> int:
        """Count sentences in text"""
        if not text:
            return 0
        
        # Split on sentence endings, but be careful with abbreviations
        sentences = re.split(r'[.!?]+\s+', text)
        # Filter out very short "sentences" that might be abbreviations
        sentences = [s for s in sentences if len(s.strip()) > 2]
        
        return len(sentences)
    
    @staticmethod
    def get_readability_score(text: str) -> float:
        """
        Calculate a simple readability score (Flesch Reading Ease approximation)
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Readability score (higher = more readable)
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        # Count sentences, words, and syllables (approximated)
        sentences = TextProcessor.count_sentences(text)
        words = len(text.split())
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Simple syllable approximation: count vowel groups
        syllables = 0
        for word in text.split():
            word = word.lower()
            syllable_count = len(re.findall(r'[aeiouy]+', word))
            if syllable_count == 0:
                syllable_count = 1
            syllables += syllable_count
        
        # Simplified Flesch Reading Ease formula
        if syllables > 0:
            score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
            return max(0, min(100, score))  # Clamp between 0-100
        
        return 50.0  # Default middle score

class DataValidator:
    """
    Data validation utilities
    """
    
    @staticmethod
    def validate_csv_structure(df: pd.DataFrame, required_columns: List[str] = None) -> Dict[str, Any]:
        """
        Validate CSV structure and return validation results
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            required_columns (List[str]): List of required column names
            
        Returns:
            Dict: Validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns)
            }
        }
        
        # Check if DataFrame is empty
        if len(df) == 0:
            results['is_valid'] = False
            results['errors'].append("DataFrame is empty")
            return results
        
        # Check for required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                results['warnings'].append(f"Missing recommended columns: {list(missing_cols)}")
        
        # Check for completely empty columns
        empty_cols = [col for col in df.columns if df[col].isna().all()]
        if empty_cols:
            results['warnings'].append(f"Completely empty columns: {empty_cols}")
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            results['warnings'].append(f"Found {duplicate_count} duplicate rows")
        
        # Check for text columns (likely review content)
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains mostly text (average length > 20 chars)
                avg_length = df[col].astype(str).str.len().mean()
                if avg_length > 20:
                    text_columns.append(col)
        
        results['info']['potential_text_columns'] = text_columns
        
        return results
    
    @staticmethod
    def validate_review_data(df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """
        Validate review data quality
        
        Args:
            df (pd.DataFrame): DataFrame with review data
            text_column (str): Name of the text column
            
        Returns:
            Dict: Validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        if text_column not in df.columns:
            results['is_valid'] = False
            results['errors'].append(f"Text column '{text_column}' not found")
            return results
        
        # Get text statistics
        text_data = df[text_column].dropna()
        
        if len(text_data) == 0:
            results['is_valid'] = False
            results['errors'].append("No valid text data found")
            return results
        
        # Calculate statistics
        text_lengths = text_data.astype(str).str.len()
        word_counts = text_data.astype(str).str.split().str.len()
        
        results['statistics'] = {
            'total_reviews': len(text_data),
            'avg_length': float(text_lengths.mean()),
            'min_length': int(text_lengths.min()),
            'max_length': int(text_lengths.max()),
            'avg_words': float(word_counts.mean()),
            'empty_reviews': int((text_lengths == 0).sum()),
            'very_short_reviews': int((text_lengths < 10).sum()),  # Less than 10 chars
            'very_long_reviews': int((text_lengths > 1000).sum())  # More than 1000 chars
        }
        
        # Warnings for data quality issues
        if results['statistics']['empty_reviews'] > 0:
            results['warnings'].append(f"{results['statistics']['empty_reviews']} empty reviews found")
        
        if results['statistics']['very_short_reviews'] > len(text_data) * 0.1:
            results['warnings'].append("High number of very short reviews (>10% under 10 characters)")
        
        if results['statistics']['very_long_reviews'] > len(text_data) * 0.05:
            results['warnings'].append("High number of very long reviews (>5% over 1000 characters)")
        
        return results

class FileHandler:
    """
    File handling utilities
    """
    
    @staticmethod
    def safe_filename(filename: str) -> str:
        """
        Generate a safe filename by removing/replacing problematic characters
        
        Args:
            filename (str): Original filename
            
        Returns:
            str: Safe filename
        """
        # Remove problematic characters
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove multiple underscores
        safe_name = re.sub(r'_+', '_', safe_name)
        
        # Remove leading/trailing underscores and dots
        safe_name = safe_name.strip('_.')
        
        # Ensure it's not empty
        if not safe_name:
            safe_name = "file"
        
        return safe_name
    
    @staticmethod
    def generate_unique_filename(base_name: str, extension: str = "", timestamp: bool = True) -> str:
        """
        Generate a unique filename with optional timestamp
        
        Args:
            base_name (str): Base name for the file
            extension (str): File extension (with or without dot)
            timestamp (bool): Whether to add timestamp
            
        Returns:
            str: Unique filename
        """
        # Clean base name
        base_name = FileHandler.safe_filename(base_name)
        
        # Add timestamp if requested
        if timestamp:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{base_name}_{timestamp_str}"
        
        # Handle extension
        if extension and not extension.startswith('.'):
            extension = f".{extension}"
        
        return f"{base_name}{extension}"
    
    @staticmethod
    def get_file_hash(file_content: bytes) -> str:
        """
        Generate MD5 hash of file content
        
        Args:
            file_content (bytes): File content as bytes
            
        Returns:
            str: MD5 hash string
        """
        return hashlib.md5(file_content).hexdigest()
    
    @staticmethod
    def load_json_safe(filepath: str) -> Optional[Dict]:
        """
        Safely load JSON file with error handling
        
        Args:
            filepath (str): Path to JSON file
            
        Returns:
            Optional[Dict]: Loaded JSON data or None if error
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON file {filepath}: {str(e)}")
            return None
    
    @staticmethod
    def save_json_safe(data: Dict, filepath: str) -> bool:
        """
        Safely save data to JSON file
        
        Args:
            data (Dict): Data to save
            filepath (str): Output file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Data saved successfully to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving JSON file {filepath}: {str(e)}")
            return False

class ModelUtils:
    """
    Utilities for model handling and inference
    """
    
    @staticmethod
    def validate_api_key(api_key: str, service: str = "huggingface") -> bool:
        """
        Validate API key format
        
        Args:
            api_key (str): API key to validate
            service (str): Service name for validation
            
        Returns:
            bool: True if format seems valid
        """
        if not api_key or not isinstance(api_key, str):
            return False
        
        api_key = api_key.strip()
        
        if service.lower() == "huggingface":
            # HuggingFace API keys typically start with "hf_"
            return api_key.startswith("hf_") and len(api_key) > 10
        
        # Generic validation: not empty, reasonable length
        return len(api_key) > 5 and api_key.replace("-", "").replace("_", "").isalnum()
    
    @staticmethod
    def estimate_processing_time(num_reviews: int, batch_size: int = 10) -> str:
        """
        Estimate processing time for reviews
        
        Args:
            num_reviews (int): Number of reviews to process
            batch_size (int): Batch size for processing
            
        Returns:
            str: Estimated time as human-readable string
        """
        # Rough estimates based on API call times
        seconds_per_review = 2  # Conservative estimate including API latency
        total_seconds = num_reviews * seconds_per_review
        
        if total_seconds < 60:
            return f"~{int(total_seconds)} seconds"
        elif total_seconds < 3600:
            minutes = int(total_seconds / 60)
            return f"~{minutes} minute{'s' if minutes != 1 else ''}"
        else:
            hours = int(total_seconds / 3600)
            minutes = int((total_seconds % 3600) / 60)
            return f"~{hours}h {minutes}m"
    
    @staticmethod
    def parse_model_response(response: str, expected_format: str = "DECISION|CONFIDENCE|REASON") -> Dict[str, Any]:
        """
        Parse model response into structured format
        
        Args:
            response (str): Raw model response
            expected_format (str): Expected response format
            
        Returns:
            Dict: Parsed response data
        """
        result = {
            'decision': None,
            'confidence': 0.5,
            'reason': 'Could not parse response',
            'raw_response': response
        }
        
        if not response:
            return result
        
        try:
            # Try to parse pipe-separated format
            parts = response.strip().split('|')
            
            if len(parts) >= 3:
                decision_part = parts[0].strip().upper()
                confidence_part = parts[1].strip()
                reason_part = '|'.join(parts[2:]).strip()  # Join remaining parts
                
                # Parse decision
                if 'VIOLATION' in decision_part:
                    result['decision'] = 'VIOLATION' in decision_part and 'NO_VIOLATION' not in decision_part
                
                # Parse confidence
                try:
                    confidence = float(confidence_part)
                    result['confidence'] = max(0.0, min(1.0, confidence))
                except ValueError:
                    pass
                
                # Store reason
                result['reason'] = reason_part if reason_part else 'No reason provided'
            
            else:
                # Try to extract information from free text
                response_lower = response.lower()
                
                # Decision extraction
                if 'violation' in response_lower:
                    result['decision'] = 'no violation' not in response_lower
                
                # Try to find confidence numbers
                confidence_match = re.search(r'(\d+\.?\d*)', response)
                if confidence_match:
                    try:
                        conf = float(confidence_match.group(1))
                        # If it's a percentage, convert to decimal
                        if conf > 1:
                            conf = conf / 100
                        result['confidence'] = max(0.0, min(1.0, conf))
                    except ValueError:
                        pass
                
                result['reason'] = response[:200] + "..." if len(response) > 200 else response
        
        except Exception as e:
            logger.error(f"Error parsing model response: {str(e)}")
            result['reason'] = f"Parse error: {str(e)}"
        
        return result

class StreamlitUtils:
    """
    Streamlit-specific utilities
    """
    
    @staticmethod
    def show_dataframe_info(df: pd.DataFrame, title: str = "Data Information") -> None:
        """
        Display DataFrame information in Streamlit
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            title (str): Title for the information section
        """
        st.subheader(title)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", len(df))
        
        with col2:
            st.metric("Columns", len(df.columns))
        
        with col3:
            memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            st.metric("Memory Usage", f"{memory_usage:.1f} MB")
        
        with col4:
            missing_data = df.isnull().sum().sum()
            st.metric("Missing Values", missing_data)
        
        # Data types
        if st.checkbox("Show Column Details", key=f"{title}_details"):
            dtypes_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': [str(dtype) for dtype in df.dtypes],
                'Non-Null Count': [df[col].count() for col in df.columns],
                'Null Count': [df[col].isnull().sum() for col in df.columns]
            })
            st.dataframe(dtypes_df, use_container_width=True)
    
    @staticmethod
    def progress_tracker(current: int, total: int, message: str = "") -> None:
        """
        Update progress bar and status message
        
        Args:
            current (int): Current progress
            total (int): Total items
            message (str): Status message
        """
        progress = current / total if total > 0 else 0
        
        if hasattr(st, '_progress_bar'):
            st._progress_bar.progress(progress)
        
        if hasattr(st, '_status_text'):
            status_msg = f"{message} ({current}/{total})" if message else f"Processing {current}/{total}"
            st._status_text.text(status_msg)
    
    @staticmethod
    def setup_progress_tracking() -> Tuple:
        """
        Set up progress tracking components
        
        Returns:
            Tuple: (progress_bar, status_text) Streamlit components
        """
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Store references for later use
        st._progress_bar = progress_bar
        st._status_text = status_text
        
        return progress_bar, status_text
    
    @staticmethod
    def cleanup_progress_tracking() -> None:
        """Clean up progress tracking components"""
        if hasattr(st, '_progress_bar'):
            delattr(st, '_progress_bar')
        
        if hasattr(st, '_status_text'):
            delattr(st, '_status_text')
    
    @staticmethod
    def download_button_with_mime(data: Any, filename: str, label: str = "Download") -> None:
        """
        Create download button with appropriate MIME type
        
        Args:
            data: Data to download
            filename (str): Filename with extension
            label (str): Button label
        """
        # Determine MIME type from file extension
        extension = os.path.splitext(filename)[1].lower()
        
        mime_types = {
            '.csv': 'text/csv',
            '.json': 'application/json',
            '.txt': 'text/plain',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.pdf': 'application/pdf'
        }
        
        mime_type = mime_types.get(extension, 'application/octet-stream')
        
        # Convert data to appropriate format
        if isinstance(data, pd.DataFrame):
            if extension == '.csv':
                data = data.to_csv(index=False)
            elif extension == '.json':
                data = data.to_json(orient='records', indent=2)
        elif isinstance(data, dict):
            if extension == '.json':
                data = json.dumps(data, indent=2, default=str)
        
        st.download_button(
            label=label,
            data=data,
            file_name=filename,
            mime=mime_type
        )

class ConfigManager:
    """
    Configuration management utilities
    """
    
    def __init__(self, config_file: str = "config.json"):
        """
        Initialize config manager
        
        Args:
            config_file (str): Path to configuration file
        """
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            'model': {
                'default_model': 'google/gemma-2-2b-it',
                'max_length': 256,
                'temperature': 0.3,
                'confidence_threshold': 0.7
            },
            'processing': {
                'batch_size': 10,
                'max_reviews_per_batch': 1000,
                'timeout_seconds': 30
            },
            'policies': {
                'advertisement': True,
                'irrelevant': True,
                'rant_without_visit': True
            },
            'ui': {
                'show_progress': True,
                'max_display_rows': 100,
                'auto_save_interval': 300  # seconds
            }
        }
        
        if os.path.exists(self.config_file):
            loaded_config = FileHandler.load_json_safe(self.config_file)
            if loaded_config:
                # Merge with defaults
                return self._merge_configs(default_config, loaded_config)
        
        return default_config
    
    def _merge_configs(self, default: Dict, loaded: Dict) -> Dict:
        """Recursively merge configuration dictionaries"""
        result = default.copy()
        
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated path
        
        Args:
            key_path (str): Dot-separated path (e.g., 'model.max_length')
            default: Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value by dot-separated path
        
        Args:
            key_path (str): Dot-separated path
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to parent
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set value
        config[keys[-1]] = value
    
    def save(self) -> bool:
        """
        Save configuration to file
        
        Returns:
            bool: True if successful
        """
        return FileHandler.save_json_safe(self.config, self.config_file)

# Global config manager instance
config_manager = ConfigManager()

def get_config(key_path: str, default: Any = None) -> Any:
    """Get configuration value (convenience function)"""
    return config_manager.get(key_path, default)

def set_config(key_path: str, value: Any) -> None:
    """Set configuration value (convenience function)"""
    config_manager.set(key_path, value)

# Logging utilities
def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """
    Setup logging configuration
    
    Args:
        log_level (str): Logging level
        log_file (str, optional): Log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers,
        force=True
    )

def log_performance(func):
    """Decorator to log function performance"""
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        logger.info(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Completed {func.__name__} in {duration:.2f}s")
            return result
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.error(f"Error in {func.__name__} after {duration:.2f}s: {str(e)}")
            raise
    
    return wrapper

# Error handling utilities
class TrustIssuesError(Exception):
    """Base exception for TrustIssues application"""
    pass

class DataProcessingError(TrustIssuesError):
    """Exception raised during data processing"""
    pass

class ModelInferenceError(TrustIssuesError):
    """Exception raised during model inference"""
    pass

class ConfigurationError(TrustIssuesError):
    """Exception raised for configuration issues"""
    pass

def handle_error(error: Exception, context: str = "") -> str:
    """
    Handle errors and return user-friendly message
    
    Args:
        error (Exception): The exception that occurred
        context (str): Context where the error occurred
        
    Returns:
        str: User-friendly error message
    """
    error_msg = str(error)
    
    # Log the full error
    logger.error(f"Error in {context}: {error_msg}", exc_info=True)
    
    # Return user-friendly message
    if isinstance(error, DataProcessingError):
        return f"Data processing error: {error_msg}"
    elif isinstance(error, ModelInferenceError):
        return f"Model inference error: {error_msg}. Please check your API key and model configuration."
    elif isinstance(error, ConfigurationError):
        return f"Configuration error: {error_msg}"
    else:
        return f"An unexpected error occurred: {error_msg}"

# Cache utilities for Streamlit
@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_text_processing(text: str, processing_type: str) -> Any:
    """Cache text processing results"""
    processor = TextProcessor()
    
    if processing_type == "clean":
        return processor.clean_text(text)
    elif processing_type == "urls":
        return processor.extract_urls(text)
    elif processing_type == "phones":
        return processor.extract_phone_numbers(text)
    elif processing_type == "emails":
        return processor.extract_emails(text)
    elif processing_type == "readability":
        return processor.get_readability_score(text)
    else:
        return None

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def cached_data_validation(df_hash: str, column_info: Dict) -> Dict:
    """Cache data validation results"""
    # This is a placeholder - in practice, you'd pass the actual DataFrame
    # The hash ensures we recalculate if data changes
    return {"cached": True, "hash": df_hash, "info": column_info}

# Constants and default values
DEFAULT_VIOLATION_POLICIES = {
    'advertisement': {
        'enabled': True,
        'confidence_threshold': 0.7,
        'keywords': ['discount', 'offer', 'deal', 'promotion', 'visit our website']
    },
    'irrelevant': {
        'enabled': True,
        'confidence_threshold': 0.6,
        'min_relevance_score': 0.3
    },
    'rant_without_visit': {
        'enabled': True,
        'confidence_threshold': 0.8,
        'visit_indicator_threshold': 1
    }
}

SUPPORTED_FILE_FORMATS = {
    '.csv': 'text/csv',
    '.json': 'application/json',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.parquet': 'application/octet-stream'
}

MODEL_PRESETS = {
    'google/gemma-2-2b-it': {
        'max_length': 256,
        'temperature': 0.3,
        'good_for': ['general', 'advertisement_detection']
    },
    'Qwen/Qwen2-7B-Instruct': {
        'max_length': 512,
        'temperature': 0.2,
        'good_for': ['complex_reasoning', 'irrelevant_detection']
    },
    'microsoft/DialoGPT-medium': {
        'max_length': 256,
        'temperature': 0.4,
        'good_for': ['conversational', 'rant_detection']
    }
}

if __name__ == "__main__":
    # Test utilities if run directly
    print("Testing TrustIssues utilities...")
    
    # Test text processing
    processor = TextProcessor()
    sample_text = "Visit our website at www.example.com or call 555-123-4567! Great deals available."
    
    print(f"Original: {sample_text}")
    print(f"Cleaned: {processor.clean_text(sample_text)}")
    print(f"URLs: {processor.extract_urls(sample_text)}")
    print(f"Phones: {processor.extract_phone_numbers(sample_text)}")
    print(f"Readability: {processor.get_readability_score(sample_text)}")
    
    # Test model utils
    response = "VIOLATION|0.85|Contains promotional content with website and phone number"
    parsed = ModelUtils.parse_model_response(response)
    print(f"Parsed response: {parsed}")
    
    print("Utilities test completed!")
