import pandas as pd
import numpy as np
import re
import spacy
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Handles data preprocessing and feature extraction for review analysis
    """
    
    def __init__(self):
        """Initialize the data processor"""
        try:
            # Try to load spaCy model, use basic processing if not available
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Using basic text processing.")
            self.nlp = None
    
    def process_data(self, data, column_mapping):
        """
        Process raw review data and extract features
        
        Args:
            data (pd.DataFrame): Raw review data
            column_mapping (dict): Mapping of standard columns to actual column names
            
        Returns:
            pd.DataFrame: Processed data with extracted features
        """
        processed_data = data.copy()
        
        # Rename columns according to mapping
        rename_dict = {}
        for standard_name, actual_name in column_mapping.items():
            if actual_name and actual_name in data.columns:
                rename_dict[actual_name] = standard_name
        
        if rename_dict:
            processed_data = processed_data.rename(columns=rename_dict)
        
        # Ensure required columns exist
        if 'review_text' not in processed_data.columns:
            raise ValueError("Review text column not found in data")
        
        # Clean review text
        processed_data['review_text_clean'] = processed_data['review_text'].apply(self._clean_text)
        
        # Extract text features
        logger.info("Extracting text features...")
        processed_data = self._extract_text_features(processed_data)
        
        # Extract metadata features
        logger.info("Extracting metadata features...")
        processed_data = self._extract_metadata_features(processed_data)
        
        # Add review ID if not present
        if 'review_id' not in processed_data.columns:
            processed_data['review_id'] = range(len(processed_data))
        
        logger.info(f"Data processing completed. Shape: {processed_data.shape}")
        return processed_data
    
    def _clean_text(self, text):
        """
        Clean and preprocess review text
        
        Args:
            text (str): Raw review text
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text.strip()
    
    def _extract_text_features(self, data):
        """
        Extract linguistic and textual features from reviews
        
        Args:
            data (pd.DataFrame): Data with review_text_clean column
            
        Returns:
            pd.DataFrame: Data with additional text features
        """
        # Basic text statistics
        data['text_length'] = data['review_text_clean'].str.len()
        data['word_count'] = data['review_text_clean'].str.split().str.len()
        data['sentence_count'] = data['review_text_clean'].str.count(r'[.!?]+') + 1
        data['avg_word_length'] = data['review_text_clean'].apply(self._avg_word_length)
        
        # Punctuation and capitalization features
        data['exclamation_count'] = data['review_text_clean'].str.count('!')
        data['question_count'] = data['review_text_clean'].str.count(r'\?')
        data['caps_ratio'] = data['review_text'].apply(self._caps_ratio)
        
        # URL and contact info detection
        data['has_url'] = data['review_text_clean'].str.contains(
            r'http[s]?://|www\.|\.com|\.org|\.net', regex=True
        )
        data['has_phone'] = data['review_text_clean'].str.contains(
            r'\d{3}[-.]?\d{3}[-.]?\d{4}|\(\d{3}\)\s*\d{3}[-.]?\d{4}', regex=True
        )
        data['has_email'] = data['review_text_clean'].str.contains(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', regex=True
        )
        
        # Sentiment indicators (basic)
        data['positive_words'] = data['review_text_clean'].apply(self._count_positive_words)
        data['negative_words'] = data['review_text_clean'].apply(self._count_negative_words)
        data['promotional_words'] = data['review_text_clean'].apply(self._count_promotional_words)
        
        # First person indicators
        data['first_person_count'] = data['review_text_clean'].str.count(
            r'\b(i|me|my|mine|myself|we|us|our|ours|ourselves)\b'
        )
        
        # Experience indicators
        data['visited_indicators'] = data['review_text_clean'].apply(self._count_visit_indicators)
        data['indirect_indicators'] = data['review_text_clean'].apply(self._count_indirect_indicators)
        
        return data
    
    def _extract_metadata_features(self, data):
        """
        Extract features from metadata columns
        
        Args:
            data (pd.DataFrame): Data with potential metadata columns
            
        Returns:
            pd.DataFrame: Data with additional metadata features
        """
        # Rating-based features
        if 'rating' in data.columns:
            data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
            data['is_extreme_rating'] = data['rating'].apply(
                lambda x: 1 if pd.notna(x) and (x <= 2 or x >= 4) else 0
            )
            data['is_middle_rating'] = data['rating'].apply(
                lambda x: 1 if pd.notna(x) and x == 3 else 0
            )
        
        # Timestamp features
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
            data['review_hour'] = data['timestamp'].dt.hour
            data['review_day_of_week'] = data['timestamp'].dt.dayofweek
            data['review_month'] = data['timestamp'].dt.month
        
        # Location-based features
        if 'location' in data.columns:
            data['location_mentioned'] = data.apply(
                lambda row: 1 if pd.notna(row.get('location')) and 
                str(row.get('location', '')).lower() in str(row.get('review_text_clean', '')).lower() 
                else 0, axis=1
            )
        
        return data
    
    def _avg_word_length(self, text):
        """Calculate average word length in text"""
        if not text or pd.isna(text):
            return 0
        words = text.split()
        if not words:
            return 0
        return sum(len(word) for word in words) / len(words)
    
    def _caps_ratio(self, text):
        """Calculate ratio of capital letters in text"""
        if not text or pd.isna(text):
            return 0
        text = str(text)
        if len(text) == 0:
            return 0
        return sum(1 for c in text if c.isupper()) / len(text)
    
    def _count_positive_words(self, text):
        """Count positive sentiment words"""
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'awesome', 'perfect', 'love', 'best', 'nice', 'beautiful',
            'delicious', 'friendly', 'helpful', 'clean', 'fresh'
        ]
        if not text:
            return 0
        return sum(1 for word in positive_words if word in text.lower())
    
    def _count_negative_words(self, text):
        """Count negative sentiment words"""
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate',
            'disgusting', 'dirty', 'rude', 'slow', 'expensive', 'cold',
            'poor', 'disappointing', 'unacceptable', 'never', 'avoid'
        ]
        if not text:
            return 0
        return sum(1 for word in negative_words if word in text.lower())
    
    def _count_promotional_words(self, text):
        """Count promotional/advertisement words"""
        promo_words = [
            'discount', 'offer', 'deal', 'sale', 'promotion', 'coupon',
            'special', 'limited', 'free', 'bonus', 'save', 'cheap',
            'visit', 'call', 'contact', 'website', 'click', 'order'
        ]
        if not text:
            return 0
        return sum(1 for word in promo_words if word in text.lower())
    
    def _count_visit_indicators(self, text):
        """Count words indicating actual visit"""
        visit_words = [
            'went', 'visited', 'came', 'arrived', 'entered', 'ordered',
            'ate', 'drank', 'tried', 'tasted', 'experienced', 'saw',
            'met', 'talked', 'walked', 'sat', 'waited', 'paid'
        ]
        if not text:
            return 0
        return sum(1 for word in visit_words if word in text.lower())
    
    def _count_indirect_indicators(self, text):
        """Count words indicating indirect knowledge"""
        indirect_words = [
            'heard', 'told', 'said', 'people say', 'reviews say',
            'apparently', 'supposedly', 'allegedly', 'rumors',
            'friends told', 'someone said', 'i think', 'maybe',
            'probably', 'seems', 'appears', 'looks like'
        ]
        if not text:
            return 0
        count = 0
        text_lower = text.lower()
        for phrase in indirect_words:
            count += text_lower.count(phrase)
        return count
    
    def get_feature_summary(self, data):
        """
        Get summary statistics of extracted features
        
        Args:
            data (pd.DataFrame): Processed data
            
        Returns:
            pd.DataFrame: Feature summary statistics
        """
        feature_cols = [col for col in data.columns if col.startswith(
            ('text_', 'word_', 'sentence_', 'avg_', 'exclamation_', 'question_',
             'caps_', 'has_', 'positive_', 'negative_', 'promotional_',
             'first_person_', 'visited_', 'indirect_', 'is_', 'review_')
        )]
        
        if feature_cols:
            return data[feature_cols].describe()
        else:
            return pd.DataFrame()
