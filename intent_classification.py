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
                    r'\b(speak|talk|connect|transfer|escalate)\s+(to|with|me\s+(to|with))\s+(an?\s+)?(agent|human|representative|someone|person|operator)\b',
                    r'\b(need|want|require)\s+(a|an|to\s+speak\s+(to|with))?\s*(agent|human|representative|real\s+person|operator)\b',
                    r'\b(get|give)\s+me\s+(a|an|to)\s+(human|agent|representative|real\s+person)\b',
                    r'\b(customer\s+service|support|live)\s+(agent|representative|person)\b',
                    r'\b(human|real\s+person|operator)\s+(please|now|needed|support)\b',
                    r'\bswitch\s+to\s+(a\s+)?(human|agent|person)\b',
                    r'\b(I|i)\s+(need|want)\s+(human|agent|representative)\b',
                    r'\blet\s+me\s+(talk|speak)\s+to\s+(a\s+)?(human|agent|person|someone)\b'
                ],
                keywords=['agent', 'human', 'representative', 'transfer', 'connect', 'speak', 'escalate', 
                         'operator', 'live agent', 'customer service', 'real person'],
                phrase_patterns=['speak to agent', 'talk to human', 'connect me', 'transfer me', 
                               'real person', 'human help', 'customer service', 'live agent', 
                               'need agent', 'want agent', 'need human', 'get me agent']
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
        """Calculate keyword matching score for a route with context awareness"""
        text_words = set(self._tokenize(text))
        if not text_words:
            return 0.0
        
        text_lower = text.lower()
        
        # For agent_request, require stronger context
        if route.name == 'agent_request':
            # Check if it contains agent-related action verbs with proper context
            agent_keywords = {'agent', 'human', 'representative', 'operator'}
            action_verbs = {'speak', 'talk', 'connect', 'transfer', 'need', 'want', 'get'}
            
            has_agent_keyword = any(kw in text_words for kw in agent_keywords)
            has_action_verb = any(verb in text_words for verb in action_verbs)
            
            # Technical terms that indicate it's NOT an agent request
            technical_terms = {'tcp', 'http', 'api', 'protocol', 'relationship', 'between', 
                             'difference', 'comparison', 'vs', 'versus'}
            has_technical = any(term in text_words for term in technical_terms)
            
            if has_technical and not (has_agent_keyword and has_action_verb):
                return 0.0  # Clearly technical question, not agent request
            
            if not (has_agent_keyword and has_action_verb):
                return 0.0  # Need both for agent request
        
        keyword_matches = sum(1 for keyword in route.keywords 
                            if keyword.lower() in text_words or keyword.lower() in text_lower)
        
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
        
        # Default to normal_qa if:
        # 1. All scores are zero (unknown query)
        # 2. The best non-normal_qa route has very low confidence
        # 3. None of the specific routes (greeting, agent_request, scheduler) have strong signals
        
        max_specific_score = max(
            scores.get('greeting', 0.0),
            scores.get('agent_request', 0.0),
            scores.get('scheduler', 0.0)
        )
        
        # If all scores are zero or very low, default to normal_qa
        if confidence < 0.05 or max_specific_score < 0.05:
            best_route = 'normal_qa'
            confidence = scores['normal_qa']
        # If best route is not normal_qa but has low confidence and normal_qa scores higher
        elif best_route != 'normal_qa' and confidence < 0.20:
            if scores['normal_qa'] >= confidence:
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
        "Explain how this works",
        "Relationship to TCP and HTTP",
        "What is the difference between TCP and HTTP?",
        "I want to talk to an agent",
        "Tell me about your API",
        "Transport Layer Security",  # Edge case: should be normal_qa
        "SSL",  # Edge case: should be normal_qa
        "Random technical term",  # Edge case: should be normal_qa
    ]
    
    print("=" * 90)
    print("CLASSIFICATION RESULTS (Using TF-IDF, Patterns, Keywords, N-grams)")
    print("=" * 90)

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
                    r'\b(speak|talk|connect|transfer|escalate)\s+(to|with|me\s+(to|with))\s+(an?\s+)?(agent|human|representative|someone|person|operator)\b',
                    r'\b(need|want|require)\s+(a|an|to\s+speak\s+(to|with))?\s*(agent|human|representative|real\s+person|operator)\b',
                    r'\b(get|give)\s+me\s+(a|an|to)\s+(human|agent|representative|real\s+person)\b',
                    r'\b(customer\s+service|support|live)\s+(agent|representative|person)\b',
                    r'\b(human|real\s+person|operator)\s+(please|now|needed|support)\b',
                    r'\bswitch\s+to\s+(a\s+)?(human|agent|person)\b',
                    r'\b(I|i)\s+(need|want)\s+(human|agent|representative)\b',
                    r'\blet\s+me\s+(talk|speak)\s+to\s+(a\s+)?(human|agent|person|someone)\b'
                ],
                keywords=['agent', 'human', 'representative', 'transfer', 'connect', 'speak', 'escalate', 
                         'operator', 'live agent', 'customer service', 'real person'],
                phrase_patterns=['speak to agent', 'talk to human', 'connect me', 'transfer me', 
                               'real person', 'human help', 'customer service', 'live agent', 
                               'need agent', 'want agent', 'need human', 'get me agent']
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
        """Calculate keyword matching score for a route with context awareness"""
        text_words = set(self._tokenize(text))
        if not text_words:
            return 0.0
        
        text_lower = text.lower()
        
        # For agent_request, require stronger context
        if route.name == 'agent_request':
            # Check if it contains agent-related action verbs with proper context
            agent_keywords = {'agent', 'human', 'representative', 'operator'}
            action_verbs = {'speak', 'talk', 'connect', 'transfer', 'need', 'want', 'get'}
            
            has_agent_keyword = any(kw in text_words for kw in agent_keywords)
            has_action_verb = any(verb in text_words for verb in action_verbs)
            
            # Technical terms that indicate it's NOT an agent request
            technical_terms = {'tcp', 'http', 'api', 'protocol', 'relationship', 'between', 
                             'difference', 'comparison', 'vs', 'versus'}
            has_technical = any(term in text_words for term in technical_terms)
            
            if has_technical and not (has_agent_keyword and has_action_verb):
                return 0.0  # Clearly technical question, not agent request
            
            if not (has_agent_keyword and has_action_verb):
                return 0.0  # Need both for agent request
        
        keyword_matches = sum(1 for keyword in route.keywords 
                            if keyword.lower() in text_words or keyword.lower() in text_lower)
        
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
        
        # Default to normal_qa if:
        # 1. All scores are zero (unknown query)
        # 2. The best non-normal_qa route has very low confidence
        # 3. None of the specific routes (greeting, agent_request, scheduler) have strong signals
        
        max_specific_score = max(
            scores.get('greeting', 0.0),
            scores.get('agent_request', 0.0),
            scores.get('scheduler', 0.0)
        )
        
        # If all scores are zero or very low, default to normal_qa
        if confidence < 0.05 or max_specific_score < 0.05:
            best_route = 'normal_qa'
            confidence = scores['normal_qa']
        # If best route is not normal_qa but has low confidence and normal_qa scores higher
        elif best_route != 'normal_qa' and confidence < 0.20:
            if scores['normal_qa'] >= confidence:
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


# # Example usage
# if __name__ == "__main__":
#     # Initialize classifier
#     print("Initializing Lightweight Semantic Route Classifier...")
#     classifier = SemanticRouteClassifier(confidence_threshold=0.60)
#     print("✓ Classifier initialized (No heavy ML models loaded!)\n")
    
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
#         "Do you accept credit cards?",
#         "Hi",
#         "Get me a real person",
#         "Schedule me in for tomorrow",
#         "Where are you located?",
#         "Yo what's up",
#         "I need human assistance",
#         "Book an appointment",
#         "Explain how this works",
#         "Relationship to TCP and HTTP",
#         "What is the difference between TCP and HTTP?",
#         "I want to talk to an agent",
#         "Tell me about your API",
#         "Transport Layer Security",  # Edge case: should be normal_qa
#         "SSL",  # Edge case: should be normal_qa
#         "Random technical term",  # Edge case: should be normal_qa
#     ]
    
#     print("=" * 90)
#     print("CLASSIFICATION RESULTS (Using TF-IDF, Patterns, Keywords, N-grams)")
#     print("=" * 90)
    
#     for query in test_queries:
#         route, confidence, detailed = classifier.classify(query, return_scores=True)
        
#         print(f"\nQuery: '{query}'")
#         print(f"→ Route: {route.upper()}")
#         print(f"→ Confidence: {confidence:.2%}")
        
#         # Show top 3 scores for comparison
#         sorted_routes = sorted(detailed.items(), key=lambda x: x[1]['ensemble'], reverse=True)
#         print(f"→ Top 3 Routes:")
#         for i, (r_name, r_scores) in enumerate(sorted_routes[:3], 1):
#             marker = "★" if r_name == route else " "
#             print(f"   {marker} {i}. {r_name:15} → {r_scores['ensemble']:.2%} "
#                   f"(TF-IDF: {r_scores['tfidf']:.3f}, Pattern: {r_scores['pattern']:.3f}, "
#                   f"Keyword: {r_scores['keyword']:.3f})")
#         print("-" * 90)
    
#     # Batch processing demo
#     print("\n" + "=" * 90)
#     print("BATCH PROCESSING DEMO")
#     print("=" * 90)
#     batch_queries = ["hi", "I need an agent", "book appointment", "what's the price?"]
#     batch_results = classifier.batch_classify(batch_queries)
#     for query, (route, conf) in zip(batch_queries, batch_results):
#         print(f"{query:30} → {route:15} (confidence: {conf:.2%})")