import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolicyDetector:
    """
    Handles policy violation detection using ML models and rule-based approaches
    """
    
    def __init__(self, model_handler):
        """
        Initialize the policy detector
        
        Args:
            model_handler: ModelHandler instance for ML-based classification
        """
        self.model_handler = model_handler
        self.policy_rules = self._load_policy_rules()
    
    def detect_violations(self, data: pd.DataFrame, policies: Dict[str, bool], 
                         batch_size: int = 10, show_progress: bool = True) -> pd.DataFrame:
        """
        Detect policy violations in review data
        
        Args:
            data (pd.DataFrame): Processed review data
            policies (Dict[str, bool]): Which policies to enforce
            batch_size (int): Batch size for processing
            show_progress (bool): Whether to show progress bar
            
        Returns:
            pd.DataFrame: Data with violation detection results
        """
        results_data = data.copy()
        
        # Initialize result columns
        if policies.get('advertisement', False):
            results_data['advertisement_violation'] = False
            results_data['advertisement_confidence'] = 0.0
            results_data['advertisement_reason'] = ""
        
        if policies.get('irrelevant', False):
            results_data['irrelevant_violation'] = False
            results_data['irrelevant_confidence'] = 0.0
            results_data['irrelevant_reason'] = ""
        
        if policies.get('rant_without_visit', False):
            results_data['rant_violation'] = False
            results_data['rant_confidence'] = 0.0
            results_data['rant_reason'] = ""
        
        # Process reviews
        reviews = data['review_text_clean'].fillna("").tolist()
        total_reviews = len(reviews)
        
        if show_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # Process in batches
        for i in range(0, total_reviews, batch_size):
            batch_end = min(i + batch_size, total_reviews)
            batch_reviews = reviews[i:batch_end]
            
            if show_progress:
                progress = (batch_end) / total_reviews
                progress_bar.progress(progress)
                status_text.text(f"Processing reviews {i+1}-{batch_end} of {total_reviews}")
            
            # Get ML-based classifications
            try:
                batch_results = self.model_handler.classify_batch(batch_reviews, batch_size=len(batch_reviews))
                
                # Update results
                for j, result in enumerate(batch_results):
                    idx = i + j
                    
                    if policies.get('advertisement', False):
                        results_data.loc[idx, 'advertisement_violation'] = result['advertisement_violation']
                        results_data.loc[idx, 'advertisement_confidence'] = result['advertisement_confidence']
                        results_data.loc[idx, 'advertisement_reason'] = result['advertisement_reason']
                    
                    if policies.get('irrelevant', False):
                        results_data.loc[idx, 'irrelevant_violation'] = result['irrelevant_violation']
                        results_data.loc[idx, 'irrelevant_confidence'] = result['irrelevant_confidence']
                        results_data.loc[idx, 'irrelevant_reason'] = result['irrelevant_reason']
                    
                    if policies.get('rant_without_visit', False):
                        results_data.loc[idx, 'rant_violation'] = result['rant_violation']
                        results_data.loc[idx, 'rant_confidence'] = result['rant_confidence']
                        results_data.loc[idx, 'rant_reason'] = result['rant_reason']
            
            except Exception as e:
                logger.error(f"Error processing batch {i}-{batch_end}: {str(e)}")
                # Continue with next batch
                continue
        
        # Apply rule-based enhancements
        if policies.get('advertisement', False):
            results_data = self._enhance_advertisement_detection(results_data)
        
        if policies.get('irrelevant', False):
            results_data = self._enhance_irrelevant_detection(results_data)
        
        if policies.get('rant_without_visit', False):
            results_data = self._enhance_rant_detection(results_data)
        
        if show_progress:
            progress_bar.progress(1.0)
            status_text.text(f"Completed processing {total_reviews} reviews!")
        
        logger.info(f"Policy detection completed for {len(results_data)} reviews")
        return results_data
    
    def _load_policy_rules(self) -> Dict:
        """Load rule-based policy detection rules"""
        return {
            'advertisement': {
                'keywords': [
                    'discount', 'offer', 'deal', 'sale', 'promotion', 'coupon',
                    'special', 'limited time', 'free', 'bonus', 'save money',
                    'visit our website', 'call us', 'contact us', 'click here',
                    'www.', '.com', '.org', '.net', 'http', 'order now'
                ],
                'patterns': [
                    r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                    r'www\.[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',  # URLs
                    r'visit\s+(our\s+)?(website|store|location)',
                    r'call\s+(us\s+)?(now|today|at)',
                    r'\d+%\s+off',  # Percentage discounts
                ]
            },
            'irrelevant': {
                'location_keywords': [
                    'restaurant', 'food', 'service', 'staff', 'waiter', 'waitress',
                    'menu', 'dish', 'meal', 'atmosphere', 'ambiance', 'decor',
                    'store', 'shop', 'cashier', 'checkout', 'product', 'item',
                    'hotel', 'room', 'bed', 'bathroom', 'lobby', 'reception'
                ],
                'irrelevant_indicators': [
                    'my phone', 'my car', 'my house', 'my job', 'my family',
                    'politics', 'government', 'election', 'president', 'news',
                    'weather today', 'traffic', 'parking', 'gas prices'
                ]
            },
            'rant_without_visit': {
                'visit_indicators': [
                    'went to', 'visited', 'came here', 'arrived at', 'walked in',
                    'sat down', 'ordered', 'ate', 'drank', 'tried', 'tasted',
                    'experienced', 'saw', 'met', 'talked to', 'waited',
                    'paid', 'left', 'my visit', 'during my', 'when I'
                ],
                'indirect_indicators': [
                    'heard about', 'people say', 'everyone says', 'reviews say',
                    'apparently', 'supposedly', 'allegedly', 'rumors',
                    'friends told me', 'someone said', 'never been there',
                    'never been here', 'havent been', 'planning to visit'
                ],
                'rant_indicators': [
                    'terrible', 'horrible', 'worst', 'awful', 'disgusting',
                    'never again', 'avoid at all costs', 'stay away',
                    'complete waste', 'total disaster', 'absolutely terrible'
                ]
            }
        }
    
    def _enhance_advertisement_detection(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhance advertisement detection with rule-based approach"""
        if 'advertisement_violation' not in data.columns:
            return data
        
        rules = self.policy_rules['advertisement']
        
        for idx, row in data.iterrows():
            text = str(row.get('review_text_clean', '')).lower()
            
            # Count promotional keywords
            keyword_count = sum(1 for keyword in rules['keywords'] if keyword in text)
            
            # Check for promotional patterns
            pattern_matches = 0
            for pattern in rules['patterns']:
                import re
                if re.search(pattern, text, re.IGNORECASE):
                    pattern_matches += 1
            
            # Check for high promotional word count from features
            promo_words = row.get('promotional_words', 0)
            has_url = row.get('has_url', False)
            has_phone = row.get('has_phone', False)
            has_email = row.get('has_email', False)
            
            # Rule-based confidence boost
            rule_score = (keyword_count * 0.2) + (pattern_matches * 0.3) + (promo_words * 0.1)
            rule_score += (0.3 if has_url else 0) + (0.2 if has_phone else 0) + (0.2 if has_email else 0)
            
            # Enhance ML prediction with rules
            ml_confidence = row.get('advertisement_confidence', 0.5)
            combined_confidence = min(1.0, ml_confidence + rule_score)
            
            # Update violation if rule-based score is high
            if rule_score > 0.5:
                data.loc[idx, 'advertisement_violation'] = True
                data.loc[idx, 'advertisement_confidence'] = combined_confidence
                data.loc[idx, 'advertisement_reason'] += f" [Rule enhancement: {rule_score:.2f}]"
        
        return data
    
    def _enhance_irrelevant_detection(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhance irrelevant content detection with rule-based approach"""
        if 'irrelevant_violation' not in data.columns:
            return data
        
        rules = self.policy_rules['irrelevant']
        
        for idx, row in data.iterrows():
            text = str(row.get('review_text_clean', '')).lower()
            
            # Count location-relevant keywords
            location_count = sum(1 for keyword in rules['location_keywords'] if keyword in text)
            
            # Count irrelevant indicators
            irrelevant_count = sum(1 for indicator in rules['irrelevant_indicators'] if indicator in text)
            
            # Check location mention feature
            location_mentioned = row.get('location_mentioned', 0)
            
            # Calculate relevance score
            relevance_score = location_count + location_mentioned
            irrelevance_score = irrelevant_count
            
            # Rule-based assessment
            is_irrelevant_by_rules = (relevance_score == 0 and len(text) > 20) or irrelevance_score > 1
            
            if is_irrelevant_by_rules:
                data.loc[idx, 'irrelevant_violation'] = True
                data.loc[idx, 'irrelevant_confidence'] = min(1.0, 
                    row.get('irrelevant_confidence', 0.5) + 0.3)
                data.loc[idx, 'irrelevant_reason'] += f" [Rules: relevance={relevance_score}, irrelevant={irrelevance_score}]"
        
        return data
    
    def _enhance_rant_detection(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhance rant detection with rule-based approach"""
        if 'rant_violation' not in data.columns:
            return data
        
        rules = self.policy_rules['rant_without_visit']
        
        for idx, row in data.iterrows():
            text = str(row.get('review_text_clean', '')).lower()
            
            # Count visit indicators
            visit_count = sum(1 for indicator in rules['visit_indicators'] if indicator in text)
            visit_count += row.get('visited_indicators', 0)
            
            # Count indirect indicators
            indirect_count = sum(1 for indicator in rules['indirect_indicators'] if indicator in text)
            indirect_count += row.get('indirect_indicators', 0)
            
            # Count rant indicators
            rant_count = sum(1 for indicator in rules['rant_indicators'] if indicator in text)
            
            # Check rating (if available)
            rating = row.get('rating', 3)
            is_extreme_negative = rating <= 2 if pd.notna(rating) else False
            
            # Rule-based rant detection
            is_rant = (
                indirect_count > 0 and 
                visit_count == 0 and 
                (rant_count > 1 or is_extreme_negative)
            )
            
            if is_rant:
                data.loc[idx, 'rant_violation'] = True
                data.loc[idx, 'rant_confidence'] = min(1.0, 
                    row.get('rant_confidence', 0.5) + 0.2)
                data.loc[idx, 'rant_reason'] += f" [Rules: visit={visit_count}, indirect={indirect_count}, rant={rant_count}]"
        
        return data
    
    def get_summary_stats(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for policy violations
        
        Args:
            results (pd.DataFrame): Results with violation predictions
            
        Returns:
            pd.DataFrame: Summary statistics
        """
        stats = []
        
        # Advertisement violations
        if 'advertisement_violation' in results.columns:
            ad_violations = results['advertisement_violation'].sum()
            ad_rate = ad_violations / len(results) * 100
            avg_confidence = results['advertisement_confidence'].mean()
            
            stats.append({
                'Policy': 'Advertisement',
                'Violations': ad_violations,
                'Rate (%)': f"{ad_rate:.1f}%",
                'Avg Confidence': f"{avg_confidence:.3f}",
                'Total Reviews': len(results)
            })
        
        # Irrelevant content violations
        if 'irrelevant_violation' in results.columns:
            irr_violations = results['irrelevant_violation'].sum()
            irr_rate = irr_violations / len(results) * 100
            avg_confidence = results['irrelevant_confidence'].mean()
            
            stats.append({
                'Policy': 'Irrelevant Content',
                'Violations': irr_violations,
                'Rate (%)': f"{irr_rate:.1f}%",
                'Avg Confidence': f"{avg_confidence:.3f}",
                'Total Reviews': len(results)
            })
        
        # Rant violations
        if 'rant_violation' in results.columns:
            rant_violations = results['rant_violation'].sum()
            rant_rate = rant_violations / len(results) * 100
            avg_confidence = results['rant_confidence'].mean()
            
            stats.append({
                'Policy': 'Rant Without Visit',
                'Violations': rant_violations,
                'Rate (%)': f"{rant_rate:.1f}%",
                'Avg Confidence': f"{avg_confidence:.3f}",
                'Total Reviews': len(results)
            })
        
        return pd.DataFrame(stats)
    
    def export_violations(self, results: pd.DataFrame, filename: str = None) -> str:
        """
        Export violation results to CSV
        
        Args:
            results (pd.DataFrame): Results with violations
            filename (str, optional): Output filename
            
        Returns:
            str: CSV content as string
        """
        # Select relevant columns
        export_cols = ['review_id', 'review_text']
        
        # Add violation columns
        violation_cols = [col for col in results.columns if 'violation' in col or 'confidence' in col or 'reason' in col]
        export_cols.extend(violation_cols)
        
        # Add metadata columns if available
        metadata_cols = ['rating', 'location', 'timestamp']
        for col in metadata_cols:
            if col in results.columns:
                export_cols.append(col)
        
        export_data = results[export_cols].copy()
        
        if filename:
            export_data.to_csv(filename, index=False)
            return filename
        else:
            return export_data.to_csv(index=False)
