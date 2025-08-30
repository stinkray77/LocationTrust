import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
import logging
from typing import Dict, List, Optional, Tuple
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Evaluator:
    """
    Handles evaluation of model predictions against ground truth labels
    """
    
    def __init__(self):
        """Initialize the evaluator"""
        self.metrics_cache = {}
    
    def calculate_metrics(self, predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            predictions (pd.DataFrame): Model predictions
            ground_truth (pd.DataFrame): True labels
            
        Returns:
            Dict: Comprehensive metrics for each violation type
        """
        metrics = {}
        
        # Merge predictions with ground truth
        merged_data = self._merge_predictions_ground_truth(predictions, ground_truth)
        
        if merged_data is None:
            logger.error("Failed to merge predictions with ground truth")
            return {}
        
        # Calculate metrics for each violation type
        violation_types = ['advertisement', 'irrelevant', 'rant']
        
        for violation_type in violation_types:
            pred_col = f"{violation_type}_violation"
            gt_col = f"{violation_type}_gt"  # Assuming ground truth columns end with _gt
            
            if pred_col in merged_data.columns and gt_col in merged_data.columns:
                type_metrics = self._calculate_binary_metrics(
                    merged_data[gt_col], 
                    merged_data[pred_col],
                    violation_type
                )
                metrics[violation_type] = type_metrics
            else:
                logger.warning(f"Missing columns for {violation_type} evaluation")
        
        # Calculate overall metrics
        if metrics:
            metrics['overall'] = self._calculate_overall_metrics(merged_data, violation_types)
        
        return metrics
    
    def _merge_predictions_ground_truth(self, predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Merge predictions with ground truth data"""
        try:
            # Try to merge on review_id if available
            if 'review_id' in predictions.columns and 'review_id' in ground_truth.columns:
                merged = pd.merge(predictions, ground_truth, on='review_id', how='inner', suffixes=('', '_gt'))
            else:
                # Assume same order and merge by index
                if len(predictions) == len(ground_truth):
                    merged = pd.concat([predictions.reset_index(drop=True), 
                                      ground_truth.reset_index(drop=True)], axis=1)
                else:
                    logger.error("Cannot merge: different lengths and no common ID column")
                    return None
            
            logger.info(f"Merged {len(merged)} records for evaluation")
            return merged
            
        except Exception as e:
            logger.error(f"Error merging data: {str(e)}")
            return None
    
    def _calculate_binary_metrics(self, y_true: pd.Series, y_pred: pd.Series, violation_type: str) -> Dict:
        """Calculate binary classification metrics"""
        try:
            # Convert to numpy arrays and handle missing values
            y_true = np.array(y_true).astype(bool)
            y_pred = np.array(y_pred).astype(bool)
            
            # Remove any NaN values
            mask = ~(pd.isna(y_true) | pd.isna(y_pred))
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            
            if len(y_true) == 0:
                logger.warning(f"No valid samples for {violation_type}")
                return self._empty_metrics()
            
            # Calculate basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )
            
            # Macro averages
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            # Additional metrics
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Try to calculate AUC if we have confidence scores
            auc_score = None
            try:
                confidence_col = f"{violation_type}_confidence"
                if confidence_col in self.current_predictions.columns:
                    confidences = self.current_predictions[confidence_col].values[mask]
                    auc_score = roc_auc_score(y_true, confidences)
            except:
                auc_score = None
            
            metrics = {
                'accuracy': float(accuracy),
                'precision': {
                    'negative': float(precision[0]) if len(precision) > 0 else 0.0,
                    'positive': float(precision[1]) if len(precision) > 1 else 0.0,
                    'macro': float(precision_macro)
                },
                'recall': {
                    'negative': float(recall[0]) if len(recall) > 0 else 0.0,
                    'positive': float(recall[1]) if len(recall) > 1 else 0.0,
                    'macro': float(recall_macro)
                },
                'f1_score': {
                    'negative': float(f1[0]) if len(f1) > 0 else 0.0,
                    'positive': float(f1[1]) if len(f1) > 1 else 0.0,
                    'macro': float(f1_macro)
                },
                'specificity': float(specificity),
                'support': {
                    'negative': int(support[0]) if len(support) > 0 else 0,
                    'positive': int(support[1]) if len(support) > 1 else 0,
                    'total': int(len(y_true))
                },
                'confusion_matrix': {
                    'true_negative': int(tn),
                    'false_positive': int(fp),
                    'false_negative': int(fn),
                    'true_positive': int(tp)
                }
            }
            
            if auc_score is not None:
                metrics['auc_score'] = float(auc_score)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {violation_type}: {str(e)}")
            return self._empty_metrics()
    
    def _calculate_overall_metrics(self, merged_data: pd.DataFrame, violation_types: List[str]) -> Dict:
        """Calculate overall performance metrics across all violation types"""
        try:
            # Create binary vectors for overall violation detection
            any_violation_pred = pd.Series(False, index=merged_data.index)
            any_violation_gt = pd.Series(False, index=merged_data.index)
            
            for violation_type in violation_types:
                pred_col = f"{violation_type}_violation"
                gt_col = f"{violation_type}_gt"
                
                if pred_col in merged_data.columns and gt_col in merged_data.columns:
                    any_violation_pred = any_violation_pred | merged_data[pred_col].fillna(False)
                    any_violation_gt = any_violation_gt | merged_data[gt_col].fillna(False)
            
            # Calculate metrics for overall violation detection
            overall_metrics = self._calculate_binary_metrics(any_violation_gt, any_violation_pred, 'overall')
            
            # Add per-type summary
            type_summary = {}
            for violation_type in violation_types:
                pred_col = f"{violation_type}_violation"
                gt_col = f"{violation_type}_gt"
                
                if pred_col in merged_data.columns and gt_col in merged_data.columns:
                    pred_count = merged_data[pred_col].sum()
                    gt_count = merged_data[gt_col].sum()
                    type_summary[violation_type] = {
                        'predicted_violations': int(pred_count),
                        'actual_violations': int(gt_count),
                        'precision': float(pred_count / max(1, pred_count)) if pred_count > 0 else 0.0
                    }
            
            overall_metrics['violation_type_summary'] = type_summary
            
            return overall_metrics
            
        except Exception as e:
            logger.error(f"Error calculating overall metrics: {str(e)}")
            return self._empty_metrics()
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            'accuracy': 0.0,
            'precision': {'negative': 0.0, 'positive': 0.0, 'macro': 0.0},
            'recall': {'negative': 0.0, 'positive': 0.0, 'macro': 0.0},
            'f1_score': {'negative': 0.0, 'positive': 0.0, 'macro': 0.0},
            'specificity': 0.0,
            'support': {'negative': 0, 'positive': 0, 'total': 0},
            'confusion_matrix': {'true_negative': 0, 'false_positive': 0, 'false_negative': 0, 'true_positive': 0}
        }
    
    def generate_pseudo_labels(self, sample_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate pseudo labels using a more advanced model or heuristics
        
        Args:
            sample_data (pd.DataFrame): Sample data to generate labels for
            
        Returns:
            pd.DataFrame: Data with pseudo labels
        """
        try:
            # This is a simplified pseudo-labeling approach
            # In practice, you might use GPT-4 or another advanced model
            
            pseudo_labels = sample_data.copy()
            
            # Generate pseudo labels based on strong indicators
            pseudo_labels['advertisement_gt'] = self._generate_advertisement_pseudo_labels(sample_data)
            pseudo_labels['irrelevant_gt'] = self._generate_irrelevant_pseudo_labels(sample_data)
            pseudo_labels['rant_gt'] = self._generate_rant_pseudo_labels(sample_data)
            
            logger.info(f"Generated pseudo labels for {len(pseudo_labels)} samples")
            return pseudo_labels
            
        except Exception as e:
            logger.error(f"Error generating pseudo labels: {str(e)}")
            return sample_data
    
    def _generate_advertisement_pseudo_labels(self, data: pd.DataFrame) -> pd.Series:
        """Generate pseudo labels for advertisement detection"""
        labels = pd.Series(False, index=data.index)
        
        for idx, row in data.iterrows():
            score = 0
            
            # Strong indicators
            if row.get('has_url', False):
                score += 3
            if row.get('has_phone', False):
                score += 2
            if row.get('has_email', False):
                score += 2
            if row.get('promotional_words', 0) > 2:
                score += 2
            
            # Text-based indicators
            text = str(row.get('review_text_clean', '')).lower()
            if 'discount' in text or 'offer' in text or 'deal' in text:
                score += 2
            if 'visit' in text and ('website' in text or 'store' in text):
                score += 1
            
            labels[idx] = score >= 4
        
        return labels
    
    def _generate_irrelevant_pseudo_labels(self, data: pd.DataFrame) -> pd.Series:
        """Generate pseudo labels for irrelevant content detection"""
        labels = pd.Series(False, index=data.index)
        
        for idx, row in data.iterrows():
            relevance_score = 0
            
            # Check for location relevance
            if row.get('location_mentioned', 0):
                relevance_score += 2
            
            # Check for business-related terms
            text = str(row.get('review_text_clean', '')).lower()
            business_terms = ['service', 'staff', 'food', 'atmosphere', 'experience', 'place']
            relevance_score += sum(1 for term in business_terms if term in text)
            
            # Check for irrelevant topics
            irrelevant_terms = ['phone', 'car', 'politics', 'weather', 'news']
            irrelevant_score = sum(1 for term in irrelevant_terms if term in text)
            
            # Label as irrelevant if low relevance and high irrelevance
            labels[idx] = (relevance_score <= 1 and irrelevant_score > 0 and len(text) > 20)
        
        return labels
    
    def _generate_rant_pseudo_labels(self, data: pd.DataFrame) -> pd.Series:
        """Generate pseudo labels for rant without visit detection"""
        labels = pd.Series(False, index=data.index)
        
        for idx, row in data.iterrows():
            visit_score = row.get('visited_indicators', 0)
            indirect_score = row.get('indirect_indicators', 0)
            negative_words = row.get('negative_words', 0)
            rating = row.get('rating', 3)
            
            # Strong indicators of rant without visit
            is_rant = (
                indirect_score > 0 and 
                visit_score == 0 and 
                negative_words > 2 and
                (pd.isna(rating) or rating <= 2)
            )
            
            labels[idx] = is_rant
        
        return labels
    
    def generate_evaluation_report(self, metrics: Dict) -> str:
        """
        Generate a comprehensive evaluation report
        
        Args:
            metrics (Dict): Calculated metrics
            
        Returns:
            str: Formatted evaluation report
        """
        report_lines = []
        report_lines.append("=== TrustIssues Model Evaluation Report ===\n")
        
        for violation_type, type_metrics in metrics.items():
            if violation_type == 'overall':
                report_lines.append("OVERALL PERFORMANCE:")
            else:
                report_lines.append(f"\n{violation_type.upper()} VIOLATION DETECTION:")
            
            report_lines.append("-" * 50)
            
            # Basic metrics
            if 'accuracy' in type_metrics:
                report_lines.append(f"Accuracy: {type_metrics['accuracy']:.3f}")
            
            if 'precision' in type_metrics:
                report_lines.append(f"Precision (Macro): {type_metrics['precision']['macro']:.3f}")
                report_lines.append(f"Recall (Macro): {type_metrics['recall']['macro']:.3f}")
                report_lines.append(f"F1-Score (Macro): {type_metrics['f1_score']['macro']:.3f}")
            
            # Confusion matrix
            if 'confusion_matrix' in type_metrics:
                cm = type_metrics['confusion_matrix']
                report_lines.append("\nConfusion Matrix:")
                report_lines.append(f"  True Positives:  {cm['true_positive']}")
                report_lines.append(f"  False Positives: {cm['false_positive']}")
                report_lines.append(f"  True Negatives:  {cm['true_negative']}")
                report_lines.append(f"  False Negatives: {cm['false_negative']}")
            
            # Support
            if 'support' in type_metrics:
                support = type_metrics['support']
                report_lines.append(f"\nSample Distribution:")
                report_lines.append(f"  Negative samples: {support['negative']}")
                report_lines.append(f"  Positive samples: {support['positive']}")
                report_lines.append(f"  Total samples: {support['total']}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_metrics(self, metrics: Dict, filepath: str) -> None:
        """
        Save metrics to JSON file
        
        Args:
            metrics (Dict): Metrics to save
            filepath (str): Output file path
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            logger.info(f"Metrics saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
