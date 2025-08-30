import os
import json
import logging
from typing import Dict, List, Optional
import requests
import time

# Try to import huggingface_hub, fallback to requests if not available
try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    InferenceClient = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelHandler:
    """
    Handles Hugging Face model inference for review classification
    """
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, 
                 max_length: int = 256, temperature: float = 0.3, 
                 confidence_threshold: float = 0.7):
        """
        Initialize the model handler
        
        Args:
            model_name (str): Name of the Hugging Face model
            api_key (str, optional): Hugging Face API key
            max_length (int): Maximum token length for generation
            temperature (float): Temperature for text generation
            confidence_threshold (float): Confidence threshold for classification
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.max_length = max_length
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        
        # Initialize the inference client
        if self.api_key and HF_HUB_AVAILABLE:
            self.client = InferenceClient(
                model=model_name,
                token=self.api_key
            )
        else:
            # Fallback to local processing if no API key or no huggingface_hub
            self.client = None
            if not HF_HUB_AVAILABLE:
                logger.warning("Hugging Face Hub not available. Using fallback classification.")
            elif not self.api_key:
                logger.warning("No API key provided. Using fallback classification.")
        
        logger.info(f"Model handler initialized for {model_name}")
    
    def classify_review(self, review_text: str) -> Dict:
        """
        Classify a single review for policy violations
        
        Args:
            review_text (str): The review text to classify
            
        Returns:
            Dict: Classification results with confidence scores
        """
        if not review_text or not review_text.strip():
            return self._empty_result()
        
        try:
            # Get classifications for each policy
            advertisement_result = self._classify_advertisement(review_text)
            irrelevant_result = self._classify_irrelevant(review_text)
            rant_result = self._classify_rant(review_text)
            
            return {
                'advertisement_violation': advertisement_result['violation'],
                'advertisement_confidence': advertisement_result['confidence'],
                'advertisement_reason': advertisement_result['reason'],
                'irrelevant_violation': irrelevant_result['violation'],
                'irrelevant_confidence': irrelevant_result['confidence'],
                'irrelevant_reason': irrelevant_result['reason'],
                'rant_violation': rant_result['violation'],
                'rant_confidence': rant_result['confidence'],
                'rant_reason': rant_result['reason'],
                'overall_trustworthy': self._calculate_overall_trust(
                    advertisement_result, irrelevant_result, rant_result
                )
            }
            
        except Exception as e:
            logger.error(f"Error classifying review: {str(e)}")
            return self._empty_result()
    
    def classify_batch(self, reviews: List[str], batch_size: int = 10) -> List[Dict]:
        """
        Classify multiple reviews in batches
        
        Args:
            reviews (List[str]): List of review texts
            batch_size (int): Number of reviews to process at once
            
        Returns:
            List[Dict]: List of classification results
        """
        results = []
        
        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(reviews)-1)//batch_size + 1}")
            
            batch_results = []
            for review in batch:
                result = self.classify_review(review)
                batch_results.append(result)
                
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
            
            results.extend(batch_results)
        
        return results
    
    def _classify_advertisement(self, review_text: str) -> Dict:
        """Classify if review contains advertisement"""
        prompt = f"""
        Analyze the following review and determine if it contains advertisement or promotional content.

        Review: "{review_text}"

        Look for:
        - Promotional language or marketing speak
        - URLs, websites, or contact information
        - Discount offers or special deals
        - Calls to action (visit, call, contact)
        - Business promotion rather than genuine review

        Respond with:
        1. "VIOLATION" if this is advertisement/promotional content
        2. "NO_VIOLATION" if this is a genuine review
        3. Confidence score (0.0-1.0)
        4. Brief reason for your decision

        Format: DECISION|CONFIDENCE|REASON
        """
        
        return self._process_classification_prompt(prompt, "advertisement")
    
    def _classify_irrelevant(self, review_text: str) -> Dict:
        """Classify if review is irrelevant to location"""
        prompt = f"""
        Analyze the following review and determine if it's relevant to the location/business being reviewed.

        Review: "{review_text}"

        Look for:
        - Content completely unrelated to the business/location
        - Personal stories not about the venue
        - Product reviews unrelated to the establishment
        - Off-topic discussions
        - Reviews about unrelated services or products

        Respond with:
        1. "VIOLATION" if this review is irrelevant to the location
        2. "NO_VIOLATION" if this review is about the location/business
        3. Confidence score (0.0-1.0)
        4. Brief reason for your decision

        Format: DECISION|CONFIDENCE|REASON
        """
        
        return self._process_classification_prompt(prompt, "irrelevant")
    
    def _classify_rant(self, review_text: str) -> Dict:
        """Classify if review is a rant from someone who hasn't visited"""
        prompt = f"""
        Analyze the following review and determine if it's a rant from someone who likely hasn't visited the location.

        Review: "{review_text}"

        Look for:
        - Extremely negative content without specific details
        - Lack of first-hand experience indicators
        - Vague complaints without concrete examples
        - Indirect language ("I heard", "people say", "apparently")
        - Disproportionate anger without visiting evidence
        - Generic complaints that could apply anywhere

        Signs of actual visit:
        - Specific details about the experience
        - Mentions of staff interactions
        - Description of food, ambiance, or services
        - Timeline of events during visit

        Respond with:
        1. "VIOLATION" if this appears to be a rant without visiting
        2. "NO_VIOLATION" if this appears to be from an actual visitor
        3. Confidence score (0.0-1.0)
        4. Brief reason for your decision

        Format: DECISION|CONFIDENCE|REASON
        """
        
        return self._process_classification_prompt(prompt, "rant")
    
    def _process_classification_prompt(self, prompt: str, classification_type: str) -> Dict:
        """Process a classification prompt and return structured result"""
        try:
            if self.client:
                # Use Hugging Face API
                response = self.client.text_generation(
                    prompt,
                    max_new_tokens=100,
                    temperature=self.temperature,
                    return_full_text=False
                )
                
                if isinstance(response, str):
                    response_text = response
                else:
                    response_text = response.generated_text if hasattr(response, 'generated_text') else str(response)
                    
            else:
                # Fallback classification
                response_text = self._fallback_classification(prompt, classification_type)
            
            # Parse the response
            return self._parse_classification_response(response_text)
            
        except Exception as e:
            logger.error(f"Error in classification prompt: {str(e)}")
            return {
                'violation': False,
                'confidence': 0.5,
                'reason': f"Error in classification: {str(e)}"
            }
    
    def _fallback_classification(self, prompt: str, classification_type: str) -> str:
        """Fallback classification using rule-based approach with improved confidence distribution"""
        # Extract the review text from prompt
        review_start = prompt.find('Review: "') + 9
        review_end = prompt.find('"', review_start)
        review_text = prompt[review_start:review_end].lower()
        
        if classification_type == "advertisement":
            promo_keywords = ['visit', 'call', 'contact', 'website', 'discount', 'offer', 'deal', 'free', 'www', '.com', 'order now', 'click here', 'special', 'promotion', 'coupon']
            strong_indicators = ['www.', '.com', 'visit our', 'call us', 'contact us', 'discount', 'free']
            
            keyword_score = sum(1 for keyword in promo_keywords if keyword in review_text)
            strong_score = sum(1 for indicator in strong_indicators if indicator in review_text)
            
            # More decisive confidence scoring
            if strong_score >= 2 or keyword_score >= 4:
                violation = True
                confidence = 0.85 + min(0.1, strong_score * 0.05)  # High confidence
            elif strong_score >= 1 or keyword_score >= 2:
                violation = True
                confidence = 0.75 + min(0.1, keyword_score * 0.02)  # Medium-high confidence
            elif keyword_score == 1:
                violation = False
                confidence = 0.25  # Low confidence, likely not ad
            else:
                violation = False
                confidence = 0.15  # Very low confidence, definitely not ad
            
            reason = f"Promotional keywords: {keyword_score}, Strong indicators: {strong_score}"
            
        elif classification_type == "irrelevant":
            location_keywords = ['food', 'service', 'staff', 'place', 'restaurant', 'store', 'shop', 'experience', 'visit', 'went', 'ordered', 'meal', 'atmosphere', 'menu', 'price', 'quality']
            irrelevant_keywords = ['politics', 'weather', 'my phone', 'my car', 'traffic', 'parking', 'government', 'news']
            
            relevant_score = sum(1 for keyword in location_keywords if keyword in review_text)
            irrelevant_score = sum(1 for keyword in irrelevant_keywords if keyword in review_text)
            
            # More decisive scoring
            if irrelevant_score >= 2 or (relevant_score == 0 and len(review_text) > 50):
                violation = True
                confidence = 0.80 + min(0.15, irrelevant_score * 0.1)  # High confidence
            elif irrelevant_score >= 1 and relevant_score <= 1:
                violation = True
                confidence = 0.70  # Medium-high confidence
            elif relevant_score >= 3:
                violation = False
                confidence = 0.20  # Low confidence, clearly relevant
            else:
                violation = False
                confidence = 0.35  # Medium-low confidence
            
            reason = f"Location terms: {relevant_score}, Irrelevant terms: {irrelevant_score}"
            
        elif classification_type == "rant":
            visit_indicators = ['went', 'visited', 'came', 'ordered', 'ate', 'tried', 'saw', 'met', 'walked in', 'sat down', 'waited', 'paid', 'left']
            indirect_indicators = ['heard', 'told', 'people say', 'apparently', 'never been', 'supposedly', 'rumors', 'friends said']
            rant_words = ['terrible', 'horrible', 'worst', 'awful', 'disgusting', 'never again', 'avoid', 'stay away', 'waste']
            
            visit_score = sum(1 for indicator in visit_indicators if indicator in review_text)
            indirect_score = sum(1 for indicator in indirect_indicators if indicator in review_text)
            rant_score = sum(1 for word in rant_words if word in review_text)
            
            # More decisive rant detection
            if indirect_score >= 2 and visit_score == 0 and rant_score >= 2:
                violation = True
                confidence = 0.88  # High confidence rant
            elif indirect_score >= 1 and visit_score == 0 and rant_score >= 1:
                violation = True
                confidence = 0.75  # Medium-high confidence
            elif visit_score >= 3:
                violation = False
                confidence = 0.15  # Low confidence, clearly visited
            elif visit_score >= 1:
                violation = False
                confidence = 0.30  # Medium-low confidence
            else:
                violation = indirect_score > visit_score
                confidence = 0.65 if violation else 0.40
            
            reason = f"Visit: {visit_score}, Indirect: {indirect_score}, Rant words: {rant_score}"
        
        else:
            violation = False
            confidence = 0.5
            reason = "Unknown classification type"
        
        decision = "VIOLATION" if violation else "NO_VIOLATION"
        return f"{decision}|{confidence:.2f}|{reason}"
    
    def _parse_classification_response(self, response_text: str) -> Dict:
        """Parse the model response into structured format"""
        try:
            # Try to parse the expected format: DECISION|CONFIDENCE|REASON
            parts = response_text.strip().split('|')
            
            if len(parts) >= 3:
                decision = parts[0].strip().upper()
                confidence = float(parts[1].strip())
                reason = parts[2].strip()
            else:
                # Try to extract information from free text
                response_lower = response_text.lower()
                if 'violation' in response_lower and 'no' not in response_lower:
                    decision = 'VIOLATION'
                    confidence = 0.7
                else:
                    decision = 'NO_VIOLATION'
                    confidence = 0.6
                reason = response_text[:100]  # First 100 chars as reason
            
            return {
                'violation': decision == 'VIOLATION',
                'confidence': max(0.0, min(1.0, confidence)),  # Clamp to [0, 1]
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return {
                'violation': False,
                'confidence': 0.5,
                'reason': f"Parse error: {response_text[:50]}..."
            }
    
    def _calculate_overall_trust(self, ad_result: Dict, irrelevant_result: Dict, rant_result: Dict) -> Dict:
        """Calculate overall trustworthiness score"""
        violations = [ad_result['violation'], irrelevant_result['violation'], rant_result['violation']]
        violation_count = sum(violations)
        
        if violation_count == 0:
            trust_score = 0.9
            trust_level = "High"
        elif violation_count == 1:
            trust_score = 0.6
            trust_level = "Medium"
        elif violation_count == 2:
            trust_score = 0.3
            trust_level = "Low"
        else:
            trust_score = 0.1
            trust_level = "Very Low"
        
        return {
            'trust_score': trust_score,
            'trust_level': trust_level,
            'violation_count': violation_count
        }
    
    def _empty_result(self) -> Dict:
        """Return empty/default classification result"""
        return {
            'advertisement_violation': False,
            'advertisement_confidence': 0.5,
            'advertisement_reason': "No classification performed",
            'irrelevant_violation': False,
            'irrelevant_confidence': 0.5,
            'irrelevant_reason': "No classification performed",
            'rant_violation': False,
            'rant_confidence': 0.5,
            'rant_reason': "No classification performed",
            'overall_trustworthy': {
                'trust_score': 0.5,
                'trust_level': "Unknown",
                'violation_count': 0
            }
        }
