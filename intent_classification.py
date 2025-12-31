# class IntentRouter:
#     """
#     Advanced intent classification with multi-signal detection.
#     Uses weighted pattern matching, sentiment analysis, and contextual rules.
#     """
    
#     # Strong signals (high confidence)
#     GREETING_STRONG = {
#         'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 
#         'good evening', 'howdy', 'hiya', 'sup'
#     }
    
#     GREETING_WEAK = {
#         'how are you', 'whats up', "what's up", 'how do you do', 'nice to meet',
#         'thanks', 'thank you', 'thx', 'ty', 'appreciate', 'bye', 'goodbye', 
#         'see you', 'take care', 'have a good', 'cheers'
#     }
    
#     AGENT_REQUEST_STRONG = {
#         'speak to agent', 'talk to agent', 'human agent', 'real person', 
#         'transfer me', 'connect me to', 'escalate', 'speak with someone',
#         'talk to someone', 'customer service', 'support team', 'representative',
#         'live agent', 'live support', 'speak to human', 'talk to human'
#     }
    
#     AGENT_REQUEST_WEAK = {
#         'human', 'agent', 'person', 'someone', 'representative', 'rep',
#         'manager', 'supervisor', 'team member', 'staff'
#     }
    
#     FRUSTRATION_SIGNALS = {
#         'frustrated', 'angry', 'upset', 'disappointed', 'not helping',
#         'useless', 'waste of time', 'ridiculous', 'terrible', 'awful',
#         'horrible', 'pathetic', 'annoying', 'sick of', 'fed up', 'enough',
#         'unacceptable', 'disgrace', 'joke', "can't believe", "won't help"
#     }
    
#     SCHEDULER_STRONG = {
#         'book appointment', 'schedule meeting', 'book demo', 'schedule demo',
#         'book a call', 'schedule call', 'set up meeting', 'arrange meeting',
#         'book time', 'reserve time', 'schedule time', 'set up call',
#         'arrange call', 'book consultation', 'schedule consultation'
#     }
    
#     SCHEDULER_WEAK = {
#         'schedule', 'book', 'appointment', 'meeting', 'demo', 'call me',
#         'callback', 'call back', 'reach out', 'contact me', 'get in touch',
#         'available', 'when can', 'calendar', 'discuss', 'consultation',
#         'convenient time', 'free time', 'talk soon'
#     }
    
#     # Question indicators (suggests QA intent)
#     QUESTION_WORDS = {
#         'what', 'when', 'where', 'why', 'how', 'who', 'which', 'whose',
#         'is', 'are', 'can', 'could', 'would', 'should', 'do', 'does',
#         'will', 'tell me', 'explain', 'describe', 'show me'
#     }
    
#     # Pricing/info keywords that override scheduler
#     INFO_REQUEST_KEYWORDS = {
#         'how much', 'cost', 'price', 'pricing', 'fee', 'charge', 'what is',
#         'tell me about', 'information about', 'details about', 'explain',
#         'describe', 'features', 'benefits', 'options', 'plans'
#     }
    
#     @classmethod
#     def _contains_phrase(cls, message: str, patterns: set) -> tuple[bool, int]:
#         """Check if message contains any patterns, return (found, match_count)"""
#         matches = 0
#         for pattern in patterns:
#             if pattern in message:
#                 matches += 1
#         return (matches > 0, matches)
    
#     @classmethod
#     def _is_question(cls, message: str) -> bool:
#         """Detect if message is a question"""
#         if '?' in message:
#             return True
#         words = message.split()
#         if len(words) > 0 and words[0] in cls.QUESTION_WORDS:
#             return True
#         return False
    
#     @classmethod
#     def _calculate_weighted_score(cls, message: str, strong_patterns: set, 
#                                    weak_patterns: set, additional_signals: set = None) -> float:
#         """Calculate weighted confidence score for an intent"""
#         score = 0.0
        
#         strong_found, strong_count = cls._contains_phrase(message, strong_patterns)
#         weak_found, weak_count = cls._contains_phrase(message, weak_patterns)
        
#         if strong_found:
#             score += 0.6 + (min(strong_count - 1, 2) * 0.15)  # 0.6 to 0.9
        
#         if weak_found:
#             score += 0.3 + (min(weak_count - 1, 2) * 0.1)  # 0.3 to 0.5
        
#         if additional_signals:
#             signal_found, signal_count = cls._contains_phrase(message, additional_signals)
#             if signal_found:
#                 score += 0.2 + (min(signal_count - 1, 1) * 0.1)  # 0.2 to 0.3
        
#         return min(score, 1.0)
    
#     @classmethod
#     def detect_intent(cls, user_message: str) -> str:
#         """
#         Detect intent using multi-signal analysis.
        
#         Returns: 'greeting', 'agent_request', 'scheduler', or 'qa'
#         """
#         message_lower = user_message.lower().strip()
#         message_len = len(message_lower.split())
        
#         # Calculate scores for each intent
#         scores = {}
        
#         # GREETING SCORE
#         scores['greeting'] = cls._calculate_weighted_score(
#             message_lower, 
#             cls.GREETING_STRONG, 
#             cls.GREETING_WEAK
#         )
#         # Boost greeting score for very short messages (1-3 words)
#         if message_len <= 3 and scores['greeting'] > 0:
#             scores['greeting'] = min(scores['greeting'] + 0.3, 1.0)
        
#         # AGENT REQUEST SCORE
#         scores['agent_request'] = cls._calculate_weighted_score(
#             message_lower,
#             cls.AGENT_REQUEST_STRONG,
#             cls.AGENT_REQUEST_WEAK,
#             cls.FRUSTRATION_SIGNALS
#         )
#         # Big boost if frustration detected
#         frustration_found, _ = cls._contains_phrase(message_lower, cls.FRUSTRATION_SIGNALS)
#         if frustration_found:
#             scores['agent_request'] = min(scores['agent_request'] + 0.3, 1.0)
        
#         # SCHEDULER SCORE
#         scores['scheduler'] = cls._calculate_weighted_score(
#             message_lower,
#             cls.SCHEDULER_STRONG,
#             cls.SCHEDULER_WEAK
#         )
#         # Penalize scheduler if it's clearly an info request
#         info_request, _ = cls._contains_phrase(message_lower, cls.INFO_REQUEST_KEYWORDS)
#         if info_request:
#             scores['scheduler'] *= 0.2  # Heavily penalize
        
#         # QA SCORE (baseline + question boost)
#         scores['qa'] = 0.4  # Base score for QA
#         if cls._is_question(message_lower):
#             scores['qa'] += 0.3
        
#         # Find highest scoring intent
#         max_intent = max(scores, key=scores.get)
#         max_score = scores[max_intent]
        
#         # Threshold check - if no strong signal, default to QA
#         if max_score < 0.4 and max_intent != 'qa':
#             return 'qa'
        
#         return max_intent
    
#     @classmethod
#     def get_confidence_score(cls, user_message: str, detected_intent: str) -> float:
#         """
#         Calculate detailed confidence score for detected intent.
#         """
#         message_lower = user_message.lower().strip()
        
#         if detected_intent == "greeting":
#             score = cls._calculate_weighted_score(
#                 message_lower,
#                 cls.GREETING_STRONG,
#                 cls.GREETING_WEAK
#             )
#         elif detected_intent == "agent_request":
#             score = cls._calculate_weighted_score(
#                 message_lower,
#                 cls.AGENT_REQUEST_STRONG,
#                 cls.AGENT_REQUEST_WEAK,
#                 cls.FRUSTRATION_SIGNALS
#             )
#         elif detected_intent == "scheduler":
#             score = cls._calculate_weighted_score(
#                 message_lower,
#                 cls.SCHEDULER_STRONG,
#                 cls.SCHEDULER_WEAK
#             )
#             # Apply info request penalty
#             info_request, _ = cls._contains_phrase(message_lower, cls.INFO_REQUEST_KEYWORDS)
#             if info_request:
#                 score *= 0.3
#         else:  # qa
#             score = 0.5
#             if cls._is_question(message_lower):
#                 score = 0.75
        
#         return min(max(score, 0.1), 1.0)  # Clamp between 0.1 and 1.0



# import re
# from typing import Dict, List, Tuple
# from dataclasses import dataclass
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# # from sentence_transformers import SentenceTransformer


# @dataclass
# class Route:
#     """Represents a route with examples and patterns"""
#     name: str
#     examples: List[str]
#     patterns: List[str]
#     keywords: List[str]


# class SemanticRouteClassifier:
#     """
#     Production-grade semantic route classifier with multiple detection strategies
#     """
    
#     def __init__(self, model_name: str = 'all-MiniLM-L6-v2', confidence_threshold: float = 0.65):
#         """
#         Initialize the classifier
        
#         Args:
#             model_name: Sentence transformer model name
#             confidence_threshold: Minimum confidence score for classification
#         """
#         self.model = SentenceTransformer(model_name)
#         self.confidence_threshold = confidence_threshold
#         self.routes = self._initialize_routes()
#         self.route_embeddings = self._compute_route_embeddings()
        
#     def _initialize_routes(self) -> Dict[str, Route]:
#         """Initialize route definitions with examples, patterns, and keywords"""
#         return {
#             'greeting': Route(
#                 name='greeting',
#                 examples=[
#                     'hello', 'hi', 'hey', 'good morning', 'good afternoon',
#                     'good evening', 'greetings', 'howdy', 'hi there',
#                     'hello there', 'hey there', 'what\'s up', 'whats up',
#                     'how are you', 'how are you doing', 'how do you do',
#                     'nice to meet you', 'pleased to meet you'
#                 ],
#                 patterns=[
#                     r'^(hi|hello|hey|greetings|howdy|yo)\b',
#                     r'\b(good\s+(morning|afternoon|evening|day))\b',
#                     r'\b(what\'?s?\s+up|how\s+are\s+you|how\s+do\s+you\s+do)\b',
#                     r'^(nice|pleased)\s+to\s+meet\s+you'
#                 ],
#                 keywords=['hi', 'hello', 'hey', 'greetings', 'morning', 'afternoon', 'evening']
#             ),
            
#             'agent_request': Route(
#                 name='agent_request',
#                 examples=[
#                     'I need to speak to an agent', 'connect me with a representative',
#                     'transfer me to a human', 'I want to talk to someone',
#                     'can I speak with an agent', 'get me a real person',
#                     'I need human assistance', 'connect me to support',
#                     'speak to customer service', 'talk to a representative',
#                     'escalate to agent', 'I need a human', 'transfer to agent',
#                     'let me talk to someone real', 'I want human help'
#                 ],
#                 patterns=[
#                     r'\b(speak|talk|connect|transfer)\s+(to|with|me\s+to)\s+(agent|human|representative|someone|person)\b',
#                     r'\b(need|want)\s+(a|an)?\s*(agent|human|representative|real\s+person)\b',
#                     r'\b(get|connect)\s+me\s+(a|to)\s+(human|agent|representative)\b',
#                     r'\bescalate\s+(to|this)\b',
#                     r'\b(customer\s+service|support)\s+(agent|representative)\b'
#                 ],
#                 keywords=['agent', 'human', 'representative', 'transfer', 'connect', 'speak', 'escalate', 'support']
#             ),
            
#             'scheduler': Route(
#                 name='scheduler',
#                 examples=[
#                     'I want to book an appointment', 'schedule a meeting',
#                     'can I make an appointment', 'book a consultation',
#                     'I need to schedule something', 'set up an appointment',
#                     'arrange a meeting', 'I want to reserve a time slot',
#                     'schedule an appointment for next week', 'book me for tomorrow',
#                     'I need to see a doctor', 'make a reservation',
#                     'I\'d like to schedule a visit', 'can I get an appointment',
#                     'set up a meeting time', 'I want to come in for an appointment'
#                 ],
#                 patterns=[
#                     r'\b(book|schedule|make|set\s+up|arrange|reserve)\s+(an?\s+)?(appointment|meeting|consultation|visit|reservation|time\s+slot)\b',
#                     r'\b(appointment|meeting|consultation|visit)\s+(booking|scheduling)\b',
#                     r'\b(need|want)\s+(to\s+)?(book|schedule|make|see)\b',
#                     r'\bcome\s+in\s+for\s+an?\s+appointment\b',
#                     r'\bget\s+an?\s+appointment\b'
#                 ],
#                 keywords=['appointment', 'schedule', 'book', 'meeting', 'consultation', 'visit', 'reservation', 'calendar']
#             ),
            
#             'normal_qa': Route(
#                 name='normal_qa',
#                 examples=[
#                     'what are your business hours', 'how much does it cost',
#                     'where are you located', 'what services do you offer',
#                     'do you accept insurance', 'what is your return policy',
#                     'how long does shipping take', 'can you help me with',
#                     'I have a question about', 'tell me about your products',
#                     'what are the requirements', 'how does this work',
#                     'explain the process', 'what options do I have'
#                 ],
#                 patterns=[
#                     r'^(what|where|when|who|why|how|which|can|do|does|is|are)\b',
#                     r'\b(tell|explain|describe|clarify)\s+me\s+(about|how)\b',
#                     r'\b(question|inquiry|wondering|curious)\s+about\b',
#                     r'\bhelp\s+me\s+(with|understand)\b'
#                 ],
#                 keywords=['what', 'how', 'where', 'when', 'why', 'question', 'help', 'information']
#             )
#         }
    
#     def _compute_route_embeddings(self) -> Dict[str, np.ndarray]:
#         """Compute embeddings for all route examples"""
#         embeddings = {}
#         for route_name, route in self.routes.items():
#             route_texts = route.examples
#             embeddings[route_name] = self.model.encode(route_texts, convert_to_numpy=True)
#         return embeddings
    
#     def _preprocess_text(self, text: str) -> str:
#         """Preprocess input text"""
#         text = text.lower().strip()
#         # Remove extra whitespace
#         text = re.sub(r'\s+', ' ', text)
#         return text
    
#     def _pattern_matching_score(self, text: str, route: Route) -> float:
#         """Calculate pattern matching score for a route"""
#         matches = 0
#         for pattern in route.patterns:
#             if re.search(pattern, text, re.IGNORECASE):
#                 matches += 1
        
#         # Normalize by number of patterns
#         return matches / len(route.patterns) if route.patterns else 0.0
    
#     def _keyword_matching_score(self, text: str, route: Route) -> float:
#         """Calculate keyword matching score for a route"""
#         text_words = set(text.split())
#         keyword_matches = sum(1 for keyword in route.keywords if keyword in text_words)
        
#         # Normalize by number of keywords
#         return keyword_matches / len(route.keywords) if route.keywords else 0.0
    
#     def _semantic_similarity_score(self, text: str, route_name: str) -> float:
#         """Calculate semantic similarity score using embeddings"""
#         query_embedding = self.model.encode([text], convert_to_numpy=True)
#         route_embeddings = self.route_embeddings[route_name]
        
#         # Calculate cosine similarity with all examples
#         similarities = cosine_similarity(query_embedding, route_embeddings)[0]
        
#         # Return max similarity score
#         return float(np.max(similarities))
    
#     def _ensemble_score(self, text: str, route_name: str) -> float:
#         """
#         Calculate ensemble score combining multiple strategies
        
#         Weights:
#         - Semantic similarity: 50%
#         - Pattern matching: 30%
#         - Keyword matching: 20%
#         """
#         route = self.routes[route_name]
        
#         semantic_score = self._semantic_similarity_score(text, route_name)
#         pattern_score = self._pattern_matching_score(text, route)
#         keyword_score = self._keyword_matching_score(text, route)
        
#         # Weighted ensemble
#         ensemble = (
#             0.50 * semantic_score +
#             0.30 * pattern_score +
#             0.20 * keyword_score
#         )
        
#         return ensemble
    
#     def classify(self, query: str, return_scores: bool = False) -> Tuple[str, float, Dict]:
#         """
#         Classify the query into one of the routes
        
#         Args:
#             query: Input query text
#             return_scores: Whether to return detailed scores
            
#         Returns:
#             Tuple of (route_name, confidence_score, detailed_scores)
#         """
#         preprocessed_query = self._preprocess_text(query)
        
#         # Calculate scores for all routes
#         scores = {}
#         detailed_scores = {}
        
#         for route_name in self.routes.keys():
#             route = self.routes[route_name]
            
#             # Get individual scores
#             semantic = self._semantic_similarity_score(preprocessed_query, route_name)
#             pattern = self._pattern_matching_score(preprocessed_query, route)
#             keyword = self._keyword_matching_score(preprocessed_query, route)
            
#             # Calculate ensemble score
#             ensemble = self._ensemble_score(preprocessed_query, route_name)
            
#             scores[route_name] = ensemble
#             detailed_scores[route_name] = {
#                 'ensemble': ensemble,
#                 'semantic': semantic,
#                 'pattern': pattern,
#                 'keyword': keyword
#             }
        
#         # Get best route
#         best_route = max(scores, key=scores.get)
#         confidence = scores[best_route]
        
#         # If confidence is below threshold, default to normal_qa
#         if confidence < self.confidence_threshold:
#             best_route = 'normal_qa'
#             confidence = scores['normal_qa']
        
#         if return_scores:
#             return best_route, confidence, detailed_scores
        
#         return best_route, confidence, detailed_scores
    
#     def batch_classify(self, queries: List[str]) -> List[Tuple[str, float]]:
#         """Classify multiple queries"""
#         results = []
#         for query in queries:
#             route, confidence, _ = self.classify(query)
#             results.append((route, confidence))
#         return results


# # Example usage
# if __name__ == "__main__":
#     # Initialize classifier
#     print("Initializing Semantic Route Classifier...")
#     classifier = SemanticRouteClassifier(confidence_threshold=0.65)
#     print("✓ Classifier initialized\n")
    
#     # Test queries
#     test_queries = [
#         "Hello, how are you today?",
#         "I need to speak with a human agent right now",
#         "Can I schedule an appointment for next Monday?",
#         "What are your business hours?",
#         "Hey there!",
#         "Transfer me to customer support",
#         "I want to book a consultation",
#         "How much does your service cost?",
#         "Good morning",
#         "Connect me with a representative please",
#         "I need to arrange a meeting",
#         "Do you accept credit cards?"
#     ]
    
#     print("=" * 80)
#     print("CLASSIFICATION RESULTS")
#     print("=" * 80)
    
#     for query in test_queries:
#         route, confidence, detailed = classifier.classify(query, return_scores=True)
        
#         print(f"\nQuery: '{query}'")
#         print(f"→ Route: {route.upper()}")
#         print(f"→ Confidence: {confidence:.2%}")
#         print(f"→ Detailed Scores:")
#         for r, scores in detailed.items():
#             if r == route:
#                 print(f"   ★ {r:15} | Ensemble: {scores['ensemble']:.3f} | "
#                       f"Semantic: {scores['semantic']:.3f} | "
#                       f"Pattern: {scores['pattern']:.3f} | "
#                       f"Keyword: {scores['keyword']:.3f}")
#         print("-" * 80)



# import re
# import requests
# from typing import Dict, List, Tuple, Optional
# from dataclasses import dataclass
# import json


# @dataclass
# class Route:
#     """Represents a route with examples and patterns"""
#     name: str
#     examples: List[str]
#     patterns: List[str]
#     keywords: List[str]


# class SemanticRouteClassifier:
#     """
#     Production-grade semantic route classifier using Ollama LLM API
#     """
    
#     def __init__(
#         self, 
#         api_url: str = 'https://aibot14.studyineurope.xyz/genaiapi/generate/',
#         confidence_threshold: float = 0.65,
#         max_tokens: int = 100,
#         temperature: float = 0.3
#     ):
#         """
#         Initialize the classifier
        
#         Args:
#             api_url: Ollama API endpoint URL
#             confidence_threshold: Minimum confidence score for classification
#             max_tokens: Maximum tokens for LLM response
#             temperature: Temperature for LLM generation
#         """
#         self.api_url = api_url
#         self.confidence_threshold = confidence_threshold
#         self.max_tokens = max_tokens
#         self.temperature = temperature
#         self.routes = self._initialize_routes()
        
#     def _initialize_routes(self) -> Dict[str, Route]:
#         """Initialize route definitions with examples, patterns, and keywords"""
#         return {
#             'greeting': Route(
#                 name='greeting',
#                 examples=[
#                     'hello', 'hi', 'hey', 'good morning', 'good afternoon',
#                     'good evening', 'greetings', 'howdy', 'hi there',
#                     'hello there', 'hey there', 'what\'s up', 'whats up',
#                     'how are you', 'how are you doing', 'how do you do',
#                     'nice to meet you', 'pleased to meet you'
#                 ],
#                 patterns=[
#                     r'^(hi|hello|hey|greetings|howdy|yo)\b',
#                     r'\b(good\s+(morning|afternoon|evening|day))\b',
#                     r'\b(what\'?s?\s+up|how\s+are\s+you|how\s+do\s+you\s+do)\b',
#                     r'^(nice|pleased)\s+to\s+meet\s+you'
#                 ],
#                 keywords=['hi', 'hello', 'hey', 'greetings', 'morning', 'afternoon', 'evening']
#             ),
            
#             'agent_request': Route(
#                 name='agent_request',
#                 examples=[
#                     'I need to speak to an agent', 'connect me with a representative',
#                     'transfer me to a human', 'I want to talk to someone',
#                     'can I speak with an agent', 'get me a real person',
#                     'I need human assistance', 'connect me to support',
#                     'speak to customer service', 'talk to a representative',
#                     'escalate to agent', 'I need a human', 'transfer to agent',
#                     'let me talk to someone real', 'I want human help'
#                 ],
#                 patterns=[
#                     r'\b(speak|talk|connect|transfer)\s+(to|with|me\s+to)\s+(agent|human|representative|someone|person)\b',
#                     r'\b(need|want)\s+(a|an)?\s*(agent|human|representative|real\s+person)\b',
#                     r'\b(get|connect)\s+me\s+(a|to)\s+(human|agent|representative)\b',
#                     r'\bescalate\s+(to|this)\b',
#                     r'\b(customer\s+service|support)\s+(agent|representative)\b'
#                 ],
#                 keywords=['agent', 'human', 'representative', 'transfer', 'connect', 'speak', 'escalate', 'support']
#             ),
            
#             'scheduler': Route(
#                 name='scheduler',
#                 examples=[
#                     'I want to book an appointment', 'schedule a meeting',
#                     'can I make an appointment', 'book a consultation',
#                     'I need to schedule something', 'set up an appointment',
#                     'arrange a meeting', 'I want to reserve a time slot',
#                     'schedule an appointment for next week', 'book me for tomorrow',
#                     'I need to see a doctor', 'make a reservation',
#                     'I\'d like to schedule a visit', 'can I get an appointment',
#                     'set up a meeting time', 'I want to come in for an appointment'
#                 ],
#                 patterns=[
#                     r'\b(book|schedule|make|set\s+up|arrange|reserve)\s+(an?\s+)?(appointment|meeting|consultation|visit|reservation|time\s+slot)\b',
#                     r'\b(appointment|meeting|consultation|visit)\s+(booking|scheduling)\b',
#                     r'\b(need|want)\s+(to\s+)?(book|schedule|make|see)\b',
#                     r'\bcome\s+in\s+for\s+an?\s+appointment\b',
#                     r'\bget\s+an?\s+appointment\b'
#                 ],
#                 keywords=['appointment', 'schedule', 'book', 'meeting', 'consultation', 'visit', 'reservation', 'calendar']
#             ),
            
#             'normal_qa': Route(
#                 name='normal_qa',
#                 examples=[
#                     'what are your business hours', 'how much does it cost',
#                     'where are you located', 'what services do you offer',
#                     'do you accept insurance', 'what is your return policy',
#                     'how long does shipping take', 'can you help me with',
#                     'I have a question about', 'tell me about your products',
#                     'what are the requirements', 'how does this work',
#                     'explain the process', 'what options do I have'
#                 ],
#                 patterns=[
#                     r'^(what|where|when|who|why|how|which|can|do|does|is|are)\b',
#                     r'\b(tell|explain|describe|clarify)\s+me\s+(about|how)\b',
#                     r'\b(question|inquiry|wondering|curious)\s+about\b',
#                     r'\bhelp\s+me\s+(with|understand)\b'
#                 ],
#                 keywords=['what', 'how', 'where', 'when', 'why', 'question', 'help', 'information']
#             )
#         }
    
#     def _preprocess_text(self, text: str) -> str:
#         """Preprocess input text"""
#         text = text.lower().strip()
#         text = re.sub(r'\s+', ' ', text)
#         return text
    
#     def _pattern_matching_score(self, text: str, route: Route) -> float:
#         """Calculate pattern matching score for a route"""
#         matches = 0
#         for pattern in route.patterns:
#             if re.search(pattern, text, re.IGNORECASE):
#                 matches += 1
#         return matches / len(route.patterns) if route.patterns else 0.0
    
#     def _keyword_matching_score(self, text: str, route: Route) -> float:
#         """Calculate keyword matching score for a route"""
#         text_words = set(text.split())
#         keyword_matches = sum(1 for keyword in route.keywords if keyword in text_words)
#         return keyword_matches / len(route.keywords) if route.keywords else 0.0
    
#     def _call_llm_api(self, prompt: str) -> Optional[str]:
#         """
#         Call Ollama LLM API to get semantic classification
        
#         Args:
#             prompt: The prompt to send to the LLM
            
#         Returns:
#             LLM response text or None if error
#         """
#         headers = {
#             'accept': 'application/json',
#             'Content-Type': 'application/json'
#         }
        
#         payload = {
#             "prompt": prompt,
#             "max_tokens": self.max_tokens,
#             "temperature": self.temperature,
#             "top_p": 0.8
#         }
        
#         try:
#             response = requests.post(
#                 self.api_url,
#                 headers=headers,
#                 json=payload,
#                 timeout=30
#             )
#             response.raise_for_status()
            
#             result = response.json()
            
#             # Handle different response formats
#             if isinstance(result, dict):
#                 return result.get('response', result.get('text', result.get('output', '')))
#             elif isinstance(result, str):
#                 return result
            
#             return None
            
#         except requests.exceptions.RequestException as e:
#             print(f"API Error: {e}")
#             return None
#         except json.JSONDecodeError as e:
#             print(f"JSON Decode Error: {e}")
#             return None
    
#     def _llm_classification_score(self, text: str, route_name: str) -> float:
#         """
#         Use LLM to calculate semantic similarity score
        
#         Args:
#             text: User query
#             route_name: Route to evaluate
            
#         Returns:
#             Similarity score between 0 and 1
#         """
#         route = self.routes[route_name]
#         examples = ', '.join(route.examples[:5])  # Use top 5 examples
        
#         prompt = f"""You are a semantic classifier. Your task is to determine how similar a user query is to a specific category.

# Category: {route_name}
# Category Examples: {examples}
# User Query: {text}

# Rate the semantic similarity between the user query and the category on a scale from 0.0 to 1.0:
# - 1.0 = Perfect match, clearly belongs to this category
# - 0.7-0.9 = Strong match, likely belongs to this category
# - 0.4-0.6 = Moderate match, might belong to this category
# - 0.1-0.3 = Weak match, unlikely to belong
# - 0.0 = No match at all

# IMPORTANT: Respond with ONLY a number between 0.0 and 1.0. No explanations, no text, just the number.

# Similarity score:"""

#         response = self._call_llm_api(prompt)
        
#         if response:
#             # Extract number from response
#             try:
#                 # Try to find a number in the response
#                 numbers = re.findall(r'\d+\.?\d*', response.strip())
#                 if numbers:
#                     score = float(numbers[0])
#                     # Ensure score is between 0 and 1
#                     if score > 1.0:
#                         score = score / 10.0 if score <= 10.0 else 1.0
#                     return min(max(score, 0.0), 1.0)
#             except (ValueError, IndexError):
#                 pass
        
#         # Fallback: use pattern and keyword matching
#         return (self._pattern_matching_score(text, route) + 
#                 self._keyword_matching_score(text, route)) / 2
    
#     def _ensemble_score(self, text: str, route_name: str) -> float:
#         """
#         Calculate ensemble score combining multiple strategies
        
#         Weights:
#         - LLM semantic similarity: 50%
#         - Pattern matching: 30%
#         - Keyword matching: 20%
#         """
#         route = self.routes[route_name]
        
#         llm_score = self._llm_classification_score(text, route_name)
#         pattern_score = self._pattern_matching_score(text, route)
#         keyword_score = self._keyword_matching_score(text, route)
        
#         ensemble = (
#             0.50 * llm_score +
#             0.30 * pattern_score +
#             0.20 * keyword_score
#         )
        
#         return ensemble
    
#     def classify(self, query: str, return_scores: bool = False) -> Tuple[str, float, Dict]:
#         """
#         Classify the query into one of the routes
        
#         Args:
#             query: Input query text
#             return_scores: Whether to return detailed scores
            
#         Returns:
#             Tuple of (route_name, confidence_score, detailed_scores)
#         """
#         preprocessed_query = self._preprocess_text(query)
        
#         scores = {}
#         detailed_scores = {}
        
#         for route_name in self.routes.keys():
#             route = self.routes[route_name]
            
#             # Get individual scores
#             llm_score = self._llm_classification_score(preprocessed_query, route_name)
#             pattern = self._pattern_matching_score(preprocessed_query, route)
#             keyword = self._keyword_matching_score(preprocessed_query, route)
            
#             # Calculate ensemble score
#             ensemble = (
#                 0.50 * llm_score +
#                 0.30 * pattern +
#                 0.20 * keyword
#             )
            
#             scores[route_name] = ensemble
#             detailed_scores[route_name] = {
#                 'ensemble': ensemble,
#                 'llm_semantic': llm_score,
#                 'pattern': pattern,
#                 'keyword': keyword
#             }
        
#         # Get best route
#         best_route = max(scores, key=scores.get)
#         confidence = scores[best_route]
        
#         # If confidence is below threshold, default to normal_qa
#         if confidence < self.confidence_threshold:
#             best_route = 'normal_qa'
#             confidence = scores['normal_qa']
        
#         if return_scores:
#             return best_route, confidence, detailed_scores
        
#         return best_route, confidence, detailed_scores
    
#     def generate_response(self, query: str, route: str) -> str:
#         """
#         Generate a response using the LLM based on the classified route
        
#         Args:
#             query: User query
#             route: Classified route name
            
#         Returns:
#             Generated response
#         """
#         # Define system prompts for each route
#         route_prompts = {
#             'greeting': f"""You are a friendly, professional customer support assistant.

# The user says: "{query}"

# Reply with a SINGLE polite greeting sentence.

# ABSOLUTE RULES:
# - ONE sentence only
# - NO explanations
# - NO meta commentary
# - NO mentioning greetings, culture, or why you responded
# - NO follow-up questions beyond offering help
# - NO code, markdown, or formatting

# Respond now with ONLY the sentence.""",

#             'agent_request': f"""You are a professional customer support assistant.

# The user says: "{query}"

# Acknowledge their request to speak with a human agent and inform them you're connecting them.

# ABSOLUTE RULES:
# - ONE to TWO sentences maximum
# - Be professional and reassuring
# - NO explanations about why they need an agent
# - NO additional questions
# - NO code, markdown, or formatting

# Respond now:""",

#             'scheduler': f"""You are a helpful scheduling assistant.

# The user says: "{query}"

# Acknowledge their appointment request and ask for their preferred date and time.

# ABSOLUTE RULES:
# - TWO sentences maximum
# - Be friendly and helpful
# - Ask for specific details needed (date/time)
# - NO explanations about the booking process
# - NO code, markdown, or formatting

# Respond now:""",

#             'normal_qa': f"""You are a knowledgeable, helpful customer support assistant.

# The user asks: "{query}"

# Provide a clear, concise answer to their question.

# ABSOLUTE RULES:
# - Be informative and direct
# - Maximum 3-4 sentences
# - NO unnecessary explanations
# - NO asking if they have more questions
# - NO code, markdown, or formatting unless specifically needed for the answer

# Respond now:"""
#         }
        
#         prompt = route_prompts.get(route, route_prompts['normal_qa'])
#         response = self._call_llm_api(prompt)
        
#         return response.strip() if response else "I apologize, but I'm having trouble generating a response. Could you please rephrase your question?"
    
#     def batch_classify(self, queries: List[str]) -> List[Tuple[str, float]]:
#         """Classify multiple queries"""
#         results = []
#         for query in queries:
#             route, confidence, _ = self.classify(query)
#             results.append((route, confidence))
#         return results


# # Example usage
# if __name__ == "__main__":
#     # Initialize classifier
#     print("Initializing Semantic Route Classifier with Ollama LLM...")
#     classifier = SemanticRouteClassifier(
#         api_url='https://aibot14.studyineurope.xyz/genaiapi/generate/',
#         confidence_threshold=0.65
#     )
#     print("✓ Classifier initialized\n")
    
#     # Test queries
#     test_queries = [
#         "Hello, how are you today?",
#         "I need to speak with a human agent right now",
#         "Can I schedule an appointment for next Monday?",
#         "What are your business hours?",
#     ]
    
#     print("=" * 80)
#     print("CLASSIFICATION & RESPONSE GENERATION")
#     print("=" * 80)
    
#     for query in test_queries:
#         print(f"\nQuery: '{query}'")
        
#         # Classify
#         route, confidence, detailed = classifier.classify(query, return_scores=True)
#         print(f"→ Route: {route.upper()}")
#         print(f"→ Confidence: {confidence:.2%}")
        
#         # Generate response
#         print(f"→ Generating response...")
#         response = classifier.generate_response(query, route)
#         print(f"→ Response: {response}")
        
#         print("-" * 80)


import re
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
import numpy as np
from collections import Counter


@dataclass
class Route:
    """Represents a route with examples and patterns"""
    name: str
    examples: List[str]
    patterns: List[str]
    keywords: List[str]
    phrase_patterns: List[str]


class SemanticRouteClassifier:
    """
    Lightweight semantic route classifier without heavy ML models
    Uses TF-IDF, N-grams, pattern matching, and keyword scoring
    """
    
    def __init__(self, confidence_threshold: float = 0.60):
        """
        Initialize the classifier
        
        Args:
            confidence_threshold: Minimum confidence score for classification
        """
        self.confidence_threshold = confidence_threshold
        self.routes = self._initialize_routes()
        self.vocabulary = self._build_vocabulary()
        self.idf_scores = self._calculate_idf()
        self.route_vectors = self._compute_route_vectors()
        
    def _initialize_routes(self) -> Dict[str, Route]:
        """Initialize route definitions with examples, patterns, and keywords"""
        return {
            'greeting': Route(
                name='greeting',
                examples=[
                    'hello', 'hi', 'hey', 'good morning', 'good afternoon',
                    'good evening', 'greetings', 'howdy', 'hi there',
                    'hello there', 'hey there', 'what\'s up', 'whats up',
                    'how are you', 'how are you doing', 'how do you do',
                    'nice to meet you', 'pleased to meet you', 'yo',
                    'hiya', 'good day', 'salutations', 'how\'s it going'
                ],
                patterns=[
                    r'^(hi|hello|hey|greetings|howdy|yo|hiya)\b',
                    r'\b(good\s+(morning|afternoon|evening|day))\b',
                    r'\b(what\'?s?\s+up|how\s+are\s+you|how\s+do\s+you\s+do)\b',
                    r'^(nice|pleased)\s+to\s+meet\s+you',
                    r'\bhow\'?s\s+it\s+going\b',
                    r'^\w{2,4}$'  # Very short greetings
                ],
                keywords=['hi', 'hello', 'hey', 'greetings', 'morning', 'afternoon', 'evening', 'howdy', 'yo'],
                phrase_patterns=['how are you', 'good morning', 'good afternoon', 'good evening', 'whats up', 'what\'s up']
            ),
            
            'agent_request': Route(
                name='agent_request',
                examples=[
                    'I need to speak to an agent', 'connect me with a representative',
                    'transfer me to a human', 'I want to talk to someone',
                    'can I speak with an agent', 'get me a real person',
                    'I need human assistance', 'connect me to support',
                    'speak to customer service', 'talk to a representative',
                    'escalate to agent', 'I need a human', 'transfer to agent',
                    'let me talk to someone real', 'I want human help',
                    'can I talk to a person', 'live agent please',
                    'human support needed', 'switch to human', 'real person please',
                    'operator please', 'I need to speak with someone',
                    'connect me to a person', 'talk to customer service'
                ],
                patterns=[
                    r'\b(speak|talk|connect|transfer)\s+(to|with|me\s+(to|with))\s+(agent|human|representative|someone|person|operator)\b',
                    r'\b(need|want|require)\s+(a|an|to\s+speak\s+to)?\s*(agent|human|representative|real\s+person|operator)\b',
                    r'\b(get|connect|transfer)\s+me\s+(a|to|with)\s+(human|agent|representative|person)\b',
                    r'\bescalate\s+(to|this|me)\b',
                    r'\b(customer\s+service|support|live)\s+(agent|representative|person)\b',
                    r'\b(human|real\s+person|operator)\s+(please|now|needed)\b',
                    r'\bswitch\s+to\s+(human|agent|person)\b'
                ],
                keywords=['agent', 'human', 'representative', 'transfer', 'connect', 'speak', 'escalate', 
                         'support', 'operator', 'live', 'person', 'real', 'customer service'],
                phrase_patterns=['speak to agent', 'talk to human', 'connect me', 'transfer me', 
                               'real person', 'human help', 'customer service', 'live agent']
            ),
            
            'scheduler': Route(
                name='scheduler',
                examples=[
                    'I want to book an appointment', 'schedule a meeting',
                    'can I make an appointment', 'book a consultation',
                    'I need to schedule something', 'set up an appointment',
                    'arrange a meeting', 'I want to reserve a time slot',
                    'schedule an appointment for next week', 'book me for tomorrow',
                    'I need to see a doctor', 'make a reservation',
                    'I\'d like to schedule a visit', 'can I get an appointment',
                    'set up a meeting time', 'I want to come in for an appointment',
                    'reserve a time', 'book a session', 'schedule a call',
                    'arrange an appointment', 'I need an appointment slot',
                    'when can I come in', 'availability for appointment',
                    'make an appointment', 'schedule me in'
                ],
                patterns=[
                    r'\b(book|schedule|make|set\s+up|arrange|reserve)\s+(an?\s+)?(appointment|meeting|consultation|visit|reservation|time\s+slot|session|call)\b',
                    r'\b(appointment|meeting|consultation|visit|session)\s+(booking|scheduling|slot)\b',
                    r'\b(need|want)\s+(to\s+)?(book|schedule|make|arrange)\b.*\b(appointment|meeting|time)\b',
                    r'\bcome\s+in\s+(for|to\s+see)\b',
                    r'\b(get|have)\s+an?\s+appointment\b',
                    r'\bavailability\s+(for|to)\b',
                    r'\bschedule\s+me\s+(in|for)\b'
                ],
                keywords=['appointment', 'schedule', 'book', 'meeting', 'consultation', 'visit', 
                         'reservation', 'calendar', 'time slot', 'reserve', 'arrange', 'session', 'availability'],
                phrase_patterns=['book appointment', 'schedule meeting', 'make appointment', 'set up meeting',
                               'reserve time', 'book session', 'schedule visit', 'appointment slot']
            ),
            
            'normal_qa': Route(
                name='normal_qa',
                examples=[
                    'what are your business hours', 'how much does it cost',
                    'where are you located', 'what services do you offer',
                    'do you accept insurance', 'what is your return policy',
                    'how long does shipping take', 'can you help me with',
                    'I have a question about', 'tell me about your products',
                    'what are the requirements', 'how does this work',
                    'explain the process', 'what options do I have',
                    'information about pricing', 'details on services',
                    'how to use this', 'where can I find', 'when do you open',
                    'why is this happening', 'which option is better'
                ],
                patterns=[
                    r'^(what|where|when|who|why|how|which|can|could|do|does|is|are|will|would)\b',
                    r'\b(tell|explain|describe|clarify|show)\s+(me\s+)?(about|how|what|why)\b',
                    r'\b(question|inquiry|wondering|curious|asking)\s+(about|regarding)\b',
                    r'\b(help|assist)\s+me\s+(with|understand|to)\b',
                    r'\b(information|details|info)\s+(about|on|regarding)\b',
                    r'\bhow\s+(do|can|does|to)\b'
                ],
                keywords=['what', 'how', 'where', 'when', 'why', 'question', 'help', 'information',
                         'explain', 'tell', 'cost', 'price', 'hours', 'services', 'policy'],
                phrase_patterns=['how much', 'what are', 'where is', 'how does', 'can you help',
                               'tell me', 'how to', 'business hours', 'what is']
            )
        }
    
    def _build_vocabulary(self) -> Dict[str, int]:
        """Build vocabulary from all route examples"""
        vocab = set()
        for route in self.routes.values():
            for example in route.examples:
                words = self._tokenize(example)
                vocab.update(words)
        return {word: idx for idx, word in enumerate(sorted(vocab))}
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        text = text.lower()
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        return [w for w in words if len(w) > 1]  # Filter very short words
    
    def _get_ngrams(self, text: str, n: int = 2) -> List[str]:
        """Extract n-grams from text"""
        words = self._tokenize(text)
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(' '.join(words[i:i+n]))
        return ngrams
    
    def _calculate_idf(self) -> Dict[str, float]:
        """Calculate IDF scores for vocabulary"""
        doc_count = {}
        total_docs = sum(len(route.examples) for route in self.routes.values())
        
        for route in self.routes.values():
            for example in route.examples:
                words = set(self._tokenize(example))
                for word in words:
                    doc_count[word] = doc_count.get(word, 0) + 1
        
        idf = {}
        for word, count in doc_count.items():
            idf[word] = np.log((total_docs + 1) / (count + 1)) + 1
        
        return idf
    
    def _text_to_tfidf_vector(self, text: str) -> np.ndarray:
        """Convert text to TF-IDF vector"""
        words = self._tokenize(text)
        word_counts = Counter(words)
        total_words = len(words) if words else 1
        
        vector = np.zeros(len(self.vocabulary))
        for word, count in word_counts.items():
            if word in self.vocabulary:
                tf = count / total_words
                idf = self.idf_scores.get(word, 1.0)
                idx = self.vocabulary[word]
                vector[idx] = tf * idf
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def _compute_route_vectors(self) -> Dict[str, np.ndarray]:
        """Compute average TF-IDF vectors for each route"""
        vectors = {}
        for route_name, route in self.routes.items():
            route_vecs = [self._text_to_tfidf_vector(ex) for ex in route.examples]
            # Average of all example vectors
            vectors[route_name] = np.mean(route_vecs, axis=0)
        return vectors
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess input text"""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def _pattern_matching_score(self, text: str, route: Route) -> float:
        """Calculate pattern matching score for a route"""
        if not route.patterns:
            return 0.0
        
        matches = 0
        for pattern in route.patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches += 1
        
        return matches / len(route.patterns)
    
    def _keyword_matching_score(self, text: str, route: Route) -> float:
        """Calculate keyword matching score for a route"""
        text_words = set(self._tokenize(text))
        if not text_words:
            return 0.0
        
        keyword_matches = sum(1 for keyword in route.keywords 
                            if keyword.lower() in text_words or keyword.lower() in text.lower())
        
        return keyword_matches / len(route.keywords) if route.keywords else 0.0
    
    def _phrase_matching_score(self, text: str, route: Route) -> float:
        """Calculate phrase matching score for a route"""
        if not route.phrase_patterns:
            return 0.0
        
        text_lower = text.lower()
        matches = sum(1 for phrase in route.phrase_patterns if phrase in text_lower)
        
        return matches / len(route.phrase_patterns)
    
    def _tfidf_similarity_score(self, text: str, route_name: str) -> float:
        """Calculate TF-IDF cosine similarity score"""
        query_vector = self._text_to_tfidf_vector(text)
        route_vector = self.route_vectors[route_name]
        
        return self._cosine_similarity(query_vector, route_vector)
    
    def _ngram_overlap_score(self, text: str, route: Route) -> float:
        """Calculate n-gram overlap score"""
        query_bigrams = set(self._get_ngrams(text, 2))
        if not query_bigrams:
            return 0.0
        
        # Get bigrams from route examples
        route_bigrams = set()
        for example in route.examples:
            route_bigrams.update(self._get_ngrams(example, 2))
        
        if not route_bigrams:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(query_bigrams & route_bigrams)
        union = len(query_bigrams | route_bigrams)
        
        return intersection / union if union > 0 else 0.0
    
    def _ensemble_score(self, text: str, route_name: str) -> float:
        """
        Calculate ensemble score combining multiple strategies
        
        Weights:
        - TF-IDF similarity: 30%
        - Pattern matching: 25%
        - Keyword matching: 20%
        - Phrase matching: 15%
        - N-gram overlap: 10%
        """
        route = self.routes[route_name]
        
        tfidf_score = self._tfidf_similarity_score(text, route_name)
        pattern_score = self._pattern_matching_score(text, route)
        keyword_score = self._keyword_matching_score(text, route)
        phrase_score = self._phrase_matching_score(text, route)
        ngram_score = self._ngram_overlap_score(text, route)
        
        ensemble = (
            0.30 * tfidf_score +
            0.25 * pattern_score +
            0.20 * keyword_score +
            0.15 * phrase_score +
            0.10 * ngram_score
        )
        
        return ensemble
    
    def classify(self, query: str, return_scores: bool = False) -> Tuple[str, float, Dict]:
        """
        Classify the query into one of the routes
        
        Args:
            query: Input query text
            return_scores: Whether to return detailed scores
            
        Returns:
            Tuple of (route_name, confidence_score, detailed_scores)
        """
        if not query or not query.strip():
            return 'normal_qa', 0.0, {}
        
        preprocessed_query = self._preprocess_text(query)
        
        scores = {}
        detailed_scores = {}
        
        for route_name in self.routes.keys():
            route = self.routes[route_name]
            
            # Get individual scores
            tfidf = self._tfidf_similarity_score(preprocessed_query, route_name)
            pattern = self._pattern_matching_score(preprocessed_query, route)
            keyword = self._keyword_matching_score(preprocessed_query, route)
            phrase = self._phrase_matching_score(preprocessed_query, route)
            ngram = self._ngram_overlap_score(preprocessed_query, route)
            
            # Calculate ensemble score
            ensemble = self._ensemble_score(preprocessed_query, route_name)
            
            scores[route_name] = ensemble
            detailed_scores[route_name] = {
                'ensemble': ensemble,
                'tfidf': tfidf,
                'pattern': pattern,
                'keyword': keyword,
                'phrase': phrase,
                'ngram': ngram
            }
        
        # Get best route based on highest score
        best_route = max(scores, key=scores.get)
        confidence = scores[best_route]
        
        # Only default to normal_qa if the confidence is extremely low AND it's not already normal_qa
        if confidence < 0.15 and best_route != 'normal_qa':
            # Check if normal_qa has a better score
            if scores['normal_qa'] > confidence:
                best_route = 'normal_qa'
                confidence = scores['normal_qa']
        
        if return_scores:
            return best_route, confidence, detailed_scores
        
        return best_route, confidence, detailed_scores
    
    def batch_classify(self, queries: List[str]) -> List[Tuple[str, float]]:
        """Classify multiple queries"""
        results = []
        for query in queries:
            route, confidence, _ = self.classify(query)
            results.append((route, confidence))
        return results


# Example usage
if __name__ == "__main__":
    # Initialize classifier
    print("Initializing Lightweight Semantic Route Classifier...")
    classifier = SemanticRouteClassifier(confidence_threshold=0.60)
    print("✓ Classifier initialized (No heavy ML models loaded!)\n")
    
    # Test queries
    test_queries = [
        "Hello, how are you today?",
        "I need to speak with a human agent right now",
        "Can I schedule an appointment for next Monday?",
        "What are your business hours?",
        "Hey there!",
        "Transfer me to customer support",
        "I want to book a consultation",
        "How much does your service cost?",
        "Good morning",
        "Connect me with a representative please",
        "I need to arrange a meeting",
        "Do you accept credit cards?",
        "Hi",
        "Get me a real person",
        "Schedule me in for tomorrow",
        "Where are you located?",
        "Yo what's up",
        "I need human assistance",
        "Book an appointment",
        "Explain how this works"
    ]
    
    print("=" * 90)
    print("CLASSIFICATION RESULTS (Using TF-IDF, Patterns, Keywords, N-grams)")
    print("=" * 90)
    
    # for query in test_queries:
    #     route, confidence, detailed = classifier.classify(query, return_scores=True)
        
    #     print(f"\nQuery: '{query}'")
    #     print(f"→ Route: {route.upper()}")
    #     print(f"→ Confidence: {confidence:.2%}")
        
    #     # Show top 3 scores for comparison
    #     sorted_routes = sorted(detailed.items(), key=lambda x: x[1]['ensemble'], reverse=True)
    #     print(f"→ Top 3 Routes:")
    #     for i, (r_name, r_scores) in enumerate(sorted_routes[:3], 1):
    #         marker = "★" if r_name == route else " "
    #         print(f"   {marker} {i}. {r_name:15} → {r_scores['ensemble']:.2%} "
    #               f"(TF-IDF: {r_scores['tfidf']:.3f}, Pattern: {r_scores['pattern']:.3f}, "
    #               f"Keyword: {r_scores['keyword']:.3f})")
    #     print("-" * 90)
    
    # Batch processing results
    # print("\n" + "=" * 90)
    # print("BATCH PROCESSING RESULTS")
    # print("=" * 90)
    # batch_queries = ["hi", "I need an agent", "book appointment", "what's the price?"]
    # batch_results = classifier.batch_classify(batch_queries)
    # for query, (route, conf) in zip(batch_queries, batch_results):
    #     print(f"{query:30} → {route:15} (confidence: {conf:.2%})")