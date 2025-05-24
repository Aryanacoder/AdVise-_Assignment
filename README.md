# AdVise-_Assignment
Identify commercial opportunities in conversations by extracting relevant keywords, intent, and product categories
Features
Topic Extraction: Identifies the most important topics using a multi-strategy approach
Intent Recognition: Calculates commercial intent on a 0-1 scale using weighted pattern detection
Product Category Matching: Maps topics to relevant product categories using a hybrid approach
Sentiment Analysis: Evaluates emotional tone with detailed explanations
Urgency Detection: Determines how pressing the expressed needs are
Conversation Analysis: Provides insights on conversational structure and context
Explainable Results: All analyses come with human-readable explanations
Technical Architecture
The system follows a pipeline architecture with five main components:
Topic Extraction: Uses multiple NLP strategies (entities, noun chunks, verb-noun compounds, etc.) with custom frequency-based scoring
Intent Recognition: Employs a weighted pattern-matching system that considers need indicators, negation, sentiment, and conversation structure
Category Matching: Implements a hybrid matching approach combining direct mapping, keyword matching, pattern detection, and contextual boosting
Sentiment Analysis: Combines TextBlob with custom emotion detection and domain-specific refinements
Result Integration: Combines all insights into a comprehensive analysis with performance metrics
