"""
Prompt templates for different violation detection tasks
"""

# Advertisement Detection Prompts
ADVERTISEMENT_PROMPT_TEMPLATE = """
You are an expert content moderator analyzing online reviews to detect advertisement and promotional content.

Review to analyze: "{review_text}"

TASK: Determine if this review contains advertisement or promotional content that violates platform policies.

CRITERIA FOR ADVERTISEMENT VIOLATIONS:
1. Contains promotional language or marketing speak
2. Includes URLs, websites, phone numbers, or email addresses
3. Mentions discount offers, special deals, or promotional codes
4. Contains calls to action (e.g., "visit", "call", "contact", "order now")
5. Focuses on promoting a business rather than genuine experience sharing
6. Uses excessive superlatives typical of marketing copy
7. Mentions competitor businesses for comparison/promotion

ANALYSIS GUIDELINES:
- Consider context: A genuine review might mention visiting a website, but promotional content actively encourages it
- Look for balance: Real reviews discuss both positives and negatives
- Check authenticity: Does this sound like a real customer experience?

RESPONSE FORMAT:
Decision: [VIOLATION or NO_VIOLATION]
Confidence: [0.0-1.0]
Reasoning: [Brief explanation of your decision]

Example responses:
- "VIOLATION|0.85|Contains promotional language and call to action with website mention"
- "NO_VIOLATION|0.72|Genuine review with authentic experience description"

Analyze the review now:
"""

IRRELEVANT_CONTENT_PROMPT_TEMPLATE = """
You are an expert content moderator analyzing online reviews to detect irrelevant content.

Review to analyze: "{review_text}"
Business/Location context: {location_context}

TASK: Determine if this review is relevant to the location/business being reviewed.

CRITERIA FOR IRRELEVANT CONTENT:
1. Discusses topics completely unrelated to the business (politics, personal life, other products)
2. Reviews a different business or location entirely
3. Contains only personal stories not related to the venue
4. Focuses on unrelated products or services not offered by the business
5. Discusses general topics without connection to the specific location
6. Contains rants about unrelated issues

RELEVANT CONTENT INDICATORS:
1. Mentions specific aspects of the business (food, service, atmosphere, products)
2. Describes actual experience at the location
3. Discusses staff interactions, wait times, prices related to the business
4. Provides feedback about the location's offerings or environment
5. Compares to similar businesses in the same category

RESPONSE FORMAT:
Decision: [VIOLATION or NO_VIOLATION]
Confidence: [0.0-1.0]
Reasoning: [Brief explanation focusing on relevance to the business]

Analyze the review now:
"""

RANT_WITHOUT_VISIT_PROMPT_TEMPLATE = """
You are an expert content moderator analyzing reviews to detect rants from users who likely never visited the location.

Review to analyze: "{review_text}"

TASK: Determine if this is a rant from someone who appears to have never actually visited the location.

INDICATORS OF ACTUAL VISIT:
1. Specific details about the experience (what they ordered, where they sat, who they interacted with)
2. Temporal markers ("when I arrived", "after waiting", "before leaving")
3. Sensory descriptions (taste, smell, sound, appearance of food/environment)
4. Staff interactions and names
5. Specific incidents with concrete details
6. Knowledge of layout, menu items, or processes that require presence

INDICATORS OF RANT WITHOUT VISIT:
1. Uses indirect language ("I heard", "people say", "apparently", "supposedly")
2. Extremely negative with no specific details
3. Generic complaints that could apply to any business
4. Mentions "never been there" or similar phrases
5. Information clearly from second-hand sources
6. Lacks any evidence of firsthand experience
7. Disproportionate anger without concrete experiences

ANALYSIS GUIDELINES:
- Focus on evidence of physical presence at the location
- Distinguish between detailed negative experiences (legitimate) and vague rants (violation)
- Consider if the reviewer provides actionable, specific feedback vs. general complaints

RESPONSE FORMAT:
Decision: [VIOLATION or NO_VIOLATION]
Confidence: [0.0-1.0]
Reasoning: [Explain presence/absence of visit indicators and specificity of complaints]

Analyze the review now:
"""

# Enhanced prompts with few-shot examples
FEW_SHOT_ADVERTISEMENT_PROMPT = """
You are an expert at detecting promotional content in reviews. Here are some examples:

EXAMPLE 1:
Review: "Great pizza! Check out our website at pizzapalace.com for 20% off your next order!"
Analysis: VIOLATION|0.95|Clear promotional content with website and discount offer

EXAMPLE 2: 
Review: "I found their website helpful for checking hours before visiting. Food was decent."
Analysis: NO_VIOLATION|0.80|Mentions website in context of genuine use, not promotion

EXAMPLE 3:
Review: "Amazing service! Call 555-1234 to book your event today. Best catering in town!"
Analysis: VIOLATION|0.90|Contains phone number and clear call to action for business promotion

Now analyze this review:
Review: "{review_text}"

Decision: [VIOLATION or NO_VIOLATION]
Confidence: [0.0-1.0]
Reasoning: [Brief explanation]
"""

# Context-aware prompts
CONTEXT_AWARE_IRRELEVANT_PROMPT = """
Business Type: {business_type}
Business Name: {business_name}
Review: "{review_text}"

As an expert moderator for {business_type} reviews, determine if this review is relevant to "{business_name}".

For a {business_type}, relevant reviews should discuss:
- Service quality and staff interactions
- {business_type}-specific aspects (food quality, atmosphere, pricing, etc.)
- Customer experience at this specific location
- Products or services offered by this type of business

Irrelevant content would include:
- Personal stories unrelated to the business
- Political or social commentary
- Reviews of different types of businesses
- General complaints not specific to the location

Analysis:
"""

# Multi-step reasoning prompts
CHAIN_OF_THOUGHT_PROMPT = """
Let's analyze this review step by step for policy violations:

Review: "{review_text}"

Step 1 - Advertisement Check:
- Does it contain promotional language? 
- Are there contact details or URLs?
- Is it encouraging business rather than sharing experience?

Step 2 - Relevance Check:
- Is it about the business/location?
- Does it discuss relevant aspects?
- Is it focused on the right topic?

Step 3 - Authenticity Check:
- Does it show evidence of actual visit?
- Are there specific, firsthand details?
- Or is it based on hearsay/assumptions?

Final Assessment:
Advertisement Violation: [YES/NO] - Confidence: [0.0-1.0]
Irrelevant Content: [YES/NO] - Confidence: [0.0-1.0] 
Rant Without Visit: [YES/NO] - Confidence: [0.0-1.0]

Reasoning for each decision:
"""

def get_advertisement_prompt(review_text: str, use_few_shot: bool = False) -> str:
    """Get advertisement detection prompt"""
    if use_few_shot:
        return FEW_SHOT_ADVERTISEMENT_PROMPT.format(review_text=review_text)
    return ADVERTISEMENT_PROMPT_TEMPLATE.format(review_text=review_text)

def get_irrelevant_prompt(review_text: str, location_context: str = "general business", 
                         business_type: str = None, business_name: str = None) -> str:
    """Get irrelevant content detection prompt"""
    if business_type and business_name:
        return CONTEXT_AWARE_IRRELEVANT_PROMPT.format(
            business_type=business_type,
            business_name=business_name,
            review_text=review_text
        )
    return IRRELEVANT_CONTENT_PROMPT_TEMPLATE.format(
        review_text=review_text, 
        location_context=location_context
    )

def get_rant_prompt(review_text: str) -> str:
    """Get rant without visit detection prompt"""
    return RANT_WITHOUT_VISIT_PROMPT_TEMPLATE.format(review_text=review_text)

def get_chain_of_thought_prompt(review_text: str) -> str:
    """Get multi-step reasoning prompt"""
    return CHAIN_OF_THOUGHT_PROMPT.format(review_text=review_text)

# Prompt optimization utilities
def optimize_prompt_for_model(base_prompt: str, model_name: str) -> str:
    """
    Optimize prompt based on specific model characteristics
    
    Args:
        base_prompt (str): Base prompt template
        model_name (str): Name of the model being used
        
    Returns:
        str: Optimized prompt
    """
    if "gemma" in model_name.lower():
        # Gemma models prefer more structured instructions
        return f"<start_of_turn>user\n{base_prompt}\n<end_of_turn>\n<start_of_turn>model\n"
    
    elif "qwen" in model_name.lower():
        # Qwen models work well with clear task definitions
        return f"Task: Content Moderation\n\n{base_prompt}\n\nResponse:"
    
    elif "llama" in model_name.lower():
        # Llama models prefer conversational format
        return f"Human: {base_prompt}\n\nAssistant:"
    
    else:
        # Default format
        return base_prompt

# Evaluation prompts for pseudo-labeling
PSEUDO_LABEL_PROMPT = """
You are a senior content moderator creating ground truth labels for training data.

Review: "{review_text}"

Provide binary labels (0 or 1) for each violation type:

1. Advertisement/Promotional (0 = legitimate review, 1 = promotional content)
2. Irrelevant Content (0 = relevant to business, 1 = off-topic/irrelevant)  
3. Rant Without Visit (0 = authentic experience, 1 = rant without visiting)

Consider these carefully - your labels will be used to train and evaluate models.

Labels:
Advertisement: [0 or 1]
Irrelevant: [0 or 1]
Rant: [0 or 1]

Confidence (0.0-1.0): [overall confidence in your labeling]
Notes: [brief reasoning for difficult cases]
"""

def get_pseudo_label_prompt(review_text: str) -> str:
    """Get pseudo-labeling prompt for ground truth generation"""
    return PSEUDO_LABEL_PROMPT.format(review_text=review_text)
