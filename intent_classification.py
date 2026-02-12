# ============================================================================
# semantic_router.py - Lightweight TF-IDF based classifier
# ============================================================================

import re
from typing import Dict, List, Tuple
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
        self.confidence_threshold = confidence_threshold
        self.routes = self._initialize_routes()
        self.vocabulary = self._build_vocabulary()
        self.idf_scores = self._calculate_idf()
        self.route_vectors = self._compute_route_vectors()
        
        # Pre-compile regex patterns for speed
        self._compiled_patterns = {}
        for route_name, route in self.routes.items():
            self._compiled_patterns[route_name] = [
                re.compile(pattern, re.IGNORECASE) for pattern in route.patterns
            ]
        
    def _initialize_routes(self) -> Dict[str, Route]:
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
                    r'^\w{2,4}$'
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
                    'connect me to a person', 'talk to customer service',
                    'i want to talk to sales', 'i want call from sales',
                    'i want to talk sales team', 'i want to talk sales',
                    'talk to sales', 'talk to sales team'
                ],
                patterns=[
                    r'\b(speak|talk|connect|transfer|escalate)\s+(to|with|me\s+(to|with))\s+(an?\s+)?(agent|human|representative|someone|person|operator)\b',
                    r'\b(need|want|require)\s+(a|an|the)?\s*(agent|human|representative|real\s+person|operator)\b',
                    r'\b(need|want)\s+to\s+(speak|talk)\s+(to|with)\s+(an?\s+)?(agent|human|representative|someone|operator)\b',
                    r'\b(get|give)\s+me\s+(a|an|to)\s+(human|agent|representative|real\s+person)\b',
                    r'\b(customer\s+service|support|live)\s+(agent|representative|person)\b',
                    r'\b(human|real\s+person|operator)\s+(please|now|needed|support)\b',
                    r'\bswitch\s+to\s+(a\s+)?(human|agent|person)\b',
                    r'\blet\s+me\s+(talk|speak)\s+to\s+(a\s+)?(human|agent|person|someone)\b',
                    r'\b(talk|speak)\s+to\s+(sales|salesperson|sales\s+team)\b',
                    r'\b(need|want)\s+(to\s+)?(talk|speak)\s+(to\s+)?(sales|salesperson|sales\s+team)\b'
                ],
                keywords=['agent', 'human', 'representative', 'transfer', 'connect', 'speak', 'escalate', 
                         'operator', 'live agent', 'customer service', 'real person',
                         'sales', 'salesperson', 'sales team'],
                phrase_patterns=['speak to agent', 'talk to human', 'connect me', 'transfer me', 
                               'real person', 'human help', 'customer service', 'live agent', 
                               'need agent', 'want agent', 'need human', 'talk to agent',
                               'talk to sales', 'talk to sales team', 'call from sales']
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
                    'set up a meeting time', 'reserve a time', 'book a session', 
                    'schedule a call', 'arrange an appointment', 'when can I come in',
                    'make an appointment', 'schedule me in',
                    'i want to schedule my meeting', 'i want to schedule a meeting',
                    'want to schedule meeting', 'schedule my meeting', 'book my appointment'
                ],
                patterns=[
                    r'\b(want|need|like|would\s+like)\s+(to\s+)?(schedule|book|arrange|set\s+up)\s+(my|a|an|the)?\s*(meeting|appointment|call|session)\b',
                    r'\b(book|schedule|make|set\s+up|arrange|reserve)\s+(an?\s+|my\s+|the\s+)?(appointment|meeting|consultation|visit|session|call)\b',
                    r'\b(appointment|meeting|consultation|visit|session)\s+(booking|scheduling|slot)\b',
                    r'\bcome\s+in\s+(for|to\s+see)\b',
                    r'\b(get|have)\s+an?\s+appointment\b',
                    r'\bavailability\s+(for|to)\b',
                    r'\bschedule\s+me\s+(in|for)\b',
                    r'\bschedule\s+(my|a|an|the)?\s*(meeting|appointment|call|session)\b',
                    r'\bbook\s+(my|a|an|the)?\s*(meeting|appointment|call|session)\b',
                ],
                keywords=['appointment', 'schedule', 'book', 'meeting', 'consultation', 'visit', 
                         'reservation', 'calendar', 'time slot', 'reserve', 'arrange', 'session', 'availability', 'call'],
                phrase_patterns=['book appointment', 'schedule meeting', 'make appointment', 'set up meeting',
                               'reserve time', 'book session', 'schedule visit', 'appointment slot',
                               'want to schedule', 'need to schedule', 'schedule my', 'book my']
            ),
            
            'conversation_close': Route(
                name='conversation_close',
                examples=[
                    'no', 'nope', 'no thanks', 'no thank you',
                    'nothing', 'nothing else', 'no i am fine',
                    'i dont need anything', 'i do not need anything',
                    'i dont need any answer', 'i do not need any answer',
                    'i dont want anything', 'i do not want anything',
                    'no need', 'not needed', 'its fine', 'it is fine',
                    'never mind', 'nevermind', 'leave it', 'forget it',
                    'thats all', 'that is all', 'nothing more',
                    'im good', 'i am good', 'im fine', 'i am fine',
                    'bye', 'goodbye', 'good bye', 'see you', 'take care',
                    'ok bye', 'okay bye', 'thanks bye', 'thank you bye',
                    'i am done', 'im done', 'done', 'all done',
                    'no more questions', 'no further questions',
                    'i dont have any more questions',
                    'thats it', 'that is it', 'enough',
                    'stop', 'end', 'close', 'exit',
                    'no i am ok', 'no its ok', 'no it is ok',
                    'not interested', 'not right now', 'maybe later',
                    'i will come back later', 'later',
                ],
                patterns=[
                    r'^(no|nope|nah)\b',
                    r'^(bye|goodbye|good\s*bye|see\s+you|take\s+care)\b',
                    r'^(done|finished|all\s+done|im\s+done|i\s+am\s+done)$',
                    r'^(nothing|nothing\s+else|no\s+need|not\s+needed)$',
                    r'^(stop|end|close|exit|leave\s+it|forget\s+it)$',
                    r'\b(dont|do\s+not|don\'t)\s+(need|want|require)\s+(any|anything|an)\b',
                    r'^(im|i\s+am)\s+(good|fine|ok|okay|done)$',
                    r'^(not\s+interested|not\s+right\s+now|maybe\s+later|later)$',
                    r'^(never\s*mind|thats\s+(all|it)|that\s+is\s+(all|it))$',
                ],
                keywords=['no', 'bye', 'goodbye', 'done', 'nothing', 'stop',
                          'end', 'leave', 'forget', 'fine', 'later', 'enough',
                          'nevermind', 'exit', 'close'],
                phrase_patterns=['no thanks', 'no thank you', 'i am fine', 'im good',
                                 'nothing else', 'thats all', 'no need', 'not needed',
                                 'leave it', 'forget it', 'never mind', 'all done',
                                 'im done', 'not interested', 'maybe later',
                                 'i dont need', 'i dont want', 'no more questions']
            ),

            'abusive': Route(
                name='abusive',
                examples=[
                    'fuck you', 'you are an idiot', 'this is bullshit',
                    'what the hell', 'piss off', 'you are useless',
                    'go to hell', 'damn it', 'you suck', 'asshole',
                    'you are a moron', 'stupid bot', 'this is garbage',
                    'kill yourself', 'i hate you', 'you are worthless'
                ],
                patterns=[
                    r'\b(fuck|shit|bitch|cunt|asshole|bastard|douchebag)\b',
                    r'\b(idiot|stupid|dumb|useless|moron|garbage)\b',
                    r'\b(hell|damn)\b',
                    r'\b(piss\s*off)\b',
                    r'\b(kill\s*yourself)\b',
                    r'\b(i\s*hate\s*you)\b'
                ],
                keywords=['fuck', 'shit', 'bitch', 'cunt', 'asshole', 'bastard', 'idiot', 'stupid', 'dumb', 'useless', 'hell', 'damn', 'piss off', 'suck', 'moron', 'garbage', 'kill', 'hate', 'worthless', 'douchebag'],
                phrase_patterns=['fuck you', 'what the hell', 'piss off', 'go to hell', 'you suck', 'you are an idiot', 'this is bullshit', 'you are useless', 'kill yourself', 'i hate you']
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
                    'how to use this', 'where can I find', 'when do you open',
                    'why is this happening', 'which option is better',
                    'i want to know about handshaking', 'i want to understand about TCP',
                    'tell me about data', 'explain HTTP protocol',
                ],
                patterns=[
                    r'^(what|where|when|who|why|how|which|can|could|do|does|is|are|will|would)\b',
                    r'\b(tell|explain|describe|clarify|show)\s+(me\s+)?(about|how|what|why)\b',
                    r'\b(question|inquiry|wondering|curious|asking)\s+(about|regarding)\b',
                    r'\b(help|assist)\s+me\s+(with|understand|to)\b',
                    r'\b(information|details|info)\s+(about|on|regarding)\b',
                    r'\bhow\s+(do|can|does|to)\b',
                    r'\b(want|need)\s+to\s+(know|understand|learn)\s+about\b',
                ],
                keywords=['what', 'how', 'where', 'when', 'why', 'question', 'help', 'information',
                         'explain', 'tell', 'cost', 'price', 'hours', 'services', 'policy', 'know', 'understand'],
                phrase_patterns=['how much', 'what are', 'where is', 'how does', 'can you help',
                               'tell me', 'how to', 'business hours', 'what is', 'want to know', 'want to understand']
            )
        }
    
    def _build_vocabulary(self) -> Dict[str, int]:
        vocab = set()
        for route in self.routes.values():
            for example in route.examples:
                words = self._tokenize(example)
                vocab.update(words)
        return {word: idx for idx, word in enumerate(sorted(vocab))}
    
    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        return [w for w in words if len(w) > 1]
    
    def _get_ngrams(self, text: str, n: int = 2) -> List[str]:
        words = self._tokenize(text)
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(' '.join(words[i:i+n]))
        return ngrams
    
    def _calculate_idf(self) -> Dict[str, float]:
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
        
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def _compute_route_vectors(self) -> Dict[str, np.ndarray]:
        vectors = {}
        for route_name, route in self.routes.items():
            route_vecs = [self._text_to_tfidf_vector(ex) for ex in route.examples]
            vectors[route_name] = np.mean(route_vecs, axis=0)
        return vectors
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _preprocess_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def _pattern_matching_score(self, text: str, route_name: str) -> float:
        if route_name not in self._compiled_patterns:
            return 0.0
        
        patterns = self._compiled_patterns[route_name]
        if not patterns:
            return 0.0
        
        if route_name == 'abusive':
            if any(pattern.search(text) for pattern in patterns):
                return 1.0

        # For agent_request, add context check
        if route_name == 'agent_request':
            text_words = set(self._tokenize(text))
            info_terms = {'know', 'understand', 'learn', 'about', 'explain', 'tell'}
            agent_keywords = {'agent', 'human', 'representative', 'operator', 'person', 'support'}
            
            has_info_term = any(term in text_words for term in info_terms)
            has_agent_keyword = any(kw in text_words for kw in agent_keywords)
            
            if has_info_term and not has_agent_keyword:
                return 0.0
        
        matches = sum(1 for pattern in patterns if pattern.search(text))
        return matches / len(patterns)
    
    def _keyword_matching_score(self, text: str, route: Route) -> float:
        text_words = set(self._tokenize(text))
        if not text_words:
            return 0.0
        
        text_lower = text.lower()
        
        if route.name == 'abusive':
            if any(keyword.lower() in text_words or keyword.lower() in text_lower for keyword in route.keywords):
                return 1.0
        
        if route.name == 'agent_request':
            agent_keywords = {'agent', 'human', 'representative', 'operator'}
            action_verbs = {'speak', 'talk', 'connect', 'transfer', 'need', 'want', 'get'}
            
            has_agent_keyword = any(kw in text_words for kw in agent_keywords)
            has_action_verb = any(verb in text_words for verb in action_verbs)
            
            technical_terms = {'tcp', 'http', 'api', 'protocol', 'relationship', 'between', 
                             'difference', 'comparison', 'vs', 'versus', 'know', 'understand', 
                             'learn', 'handshaking', 'data'}
            has_technical = any(term in text_words for term in technical_terms)
            
            if has_technical and not (has_agent_keyword and has_action_verb):
                return 0.0
            
            if not (has_agent_keyword and has_action_verb):
                return 0.0
        
        keyword_matches = sum(1 for keyword in route.keywords 
                            if keyword.lower() in text_words or keyword.lower() in text_lower)
        
        return keyword_matches / len(route.keywords) if route.keywords else 0.0
    
    def _phrase_matching_score(self, text: str, route: Route) -> float:
        if not route.phrase_patterns:
            return 0.0
        
        text_lower = text.lower()
        
        if route.name == 'abusive':
            if any(phrase in text_lower for phrase in route.phrase_patterns):
                return 1.0
        
        matches = sum(1 for phrase in route.phrase_patterns if phrase in text_lower)
        
        return matches / len(route.phrase_patterns)
    
    def _tfidf_similarity_score(self, text: str, route_name: str) -> float:
        query_vector = self._text_to_tfidf_vector(text)
        route_vector = self.route_vectors[route_name]
        
        return self._cosine_similarity(query_vector, route_vector)
    
    def _ngram_overlap_score(self, text: str, route: Route) -> float:
        query_bigrams = set(self._get_ngrams(text, 2))
        if not query_bigrams:
            return 0.0
        
        route_bigrams = set()
        for example in route.examples:
            route_bigrams.update(self._get_ngrams(example, 2))
        
        if not route_bigrams:
            return 0.0
        
        intersection = len(query_bigrams & route_bigrams)
        union = len(query_bigrams | route_bigrams)
        
        return intersection / union if union > 0 else 0.0
    
    def _ensemble_score(self, text: str, route_name: str) -> float:
        route = self.routes[route_name]
        
        tfidf_score = self._tfidf_similarity_score(text, route_name)
        pattern_score = self._pattern_matching_score(text, route_name)
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
        if not query or not query.strip():
            return 'normal_qa', 0.0, {}
        
        preprocessed_query = self._preprocess_text(query)

        # ── Early exact-match for very short conversation-close phrases ──
        _close_exact = {
            'no', 'nope', 'nah', 'bye', 'goodbye', 'done', 'nothing',
            'stop', 'end', 'exit', 'close', 'later', 'enough',
            'ok bye', 'okay bye', 'thanks bye', 'no thanks',
            'no thank you', 'im fine', 'i am fine', 'im good',
            'i am good', 'im done', 'i am done', 'all done',
            'thats all', 'that is all', 'thats it', 'that is it',
            'nothing else', 'no need', 'not needed', 'never mind',
            'nevermind', 'leave it', 'forget it', 'not interested',
            'not right now', 'maybe later', 'no i am ok', 'no its ok',
            'no it is ok', 'no more questions',
            'i dont need any answer', 'i do not need any answer',
            'i dont need anything', 'i do not need anything',
            'i dont want anything', 'i do not want anything',
        }
        if preprocessed_query in _close_exact:
            if return_scores:
                return 'conversation_close', 0.98, {'conversation_close': {'ensemble': 0.98, 'method': 'exact_match'}}
            return 'conversation_close', 0.98, {}
        
        scores = {}
        detailed_scores = {}
        
        for route_name in self.routes.keys():
            route = self.routes[route_name]
            
            tfidf = self._tfidf_similarity_score(preprocessed_query, route_name)
            pattern = self._pattern_matching_score(preprocessed_query, route_name)
            keyword = self._keyword_matching_score(preprocessed_query, route)
            phrase = self._phrase_matching_score(preprocessed_query, route)
            ngram = self._ngram_overlap_score(preprocessed_query, route)
            
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
        
        best_route = max(scores, key=scores.get)
        confidence = scores[best_route]

        # If TF-IDF thinks this is an agent_request, require BOTH
        # an agent/human marker AND a communication verb. Without both,
        # demote to normal_qa so "I want sales services" stays as QA.
        if best_route == 'agent_request':
            words = set(self._tokenize(preprocessed_query))
            agent_markers = {
                'agent', 'human', 'representative', 'operator', 'someone',
                'person', 'support', 'customer', 'service', 'sales', 'team'
            }
            comm_verbs = {
                'talk', 'speak', 'call', 'connect', 'contact',
                'reach', 'transfer', 'escalate'
            }
            has_marker = any(k in words for k in agent_markers)
            has_verb = any(v in words for v in comm_verbs)
            if not (has_marker and has_verb):
                best_route = 'normal_qa'
                confidence = scores['normal_qa']

        # If TF-IDF thinks this is scheduler, require at least one
        # explicit scheduling/booking word.  Without it, demote to
        # normal_qa so "How can I place an order?" stays as QA.
        if best_route == 'scheduler':
            words = set(self._tokenize(preprocessed_query))
            schedule_markers = {
                'schedule', 'scheduling', 'scheduled', 'reschedule',
                'book', 'booking', 'appointment', 'reservation',
                'reserve', 'slot', 'arrange', 'calendar',
                'meeting',  # only when it means "arrange a meeting"
            }
            if not any(k in words for k in schedule_markers):
                best_route = 'normal_qa'
                confidence = scores['normal_qa']

        max_specific_score = max(
            scores.get('greeting', 0.0),
            scores.get('agent_request', 0.0),
            scores.get('scheduler', 0.0),
            scores.get('conversation_close', 0.0)
        )
        
        if confidence < 0.05 or max_specific_score < 0.05:
            best_route = 'normal_qa'
            confidence = scores['normal_qa']
        elif best_route != 'normal_qa' and confidence < 0.20:
            if scores['normal_qa'] >= confidence:
                best_route = 'normal_qa'
                confidence = scores['normal_qa']
        
        if return_scores:
            return best_route, confidence, detailed_scores
        
        return best_route, confidence, detailed_scores


# ============================================================================
# bert_classifier.py - High-accuracy BERT classifier
# ============================================================================

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("⚠️  transformers/torch not installed. BERT classifier disabled.")
    print("   Install with: pip install transformers torch sentence-transformers")


class BERTIntentClassifier:
    """High-accuracy intent classifier using BERT embeddings"""
    
    def __init__(
        self, 
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        confidence_threshold: float = 0.60
    ):
        if not BERT_AVAILABLE:
            raise ImportError("Please install: pip install transformers torch")
        
        print(f"Loading BERT model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        self.confidence_threshold = confidence_threshold
        self.routes = self._initialize_routes()
        self.route_embeddings = self._compute_route_embeddings()
        print("✓ BERT model loaded!\n")
    
    def _initialize_routes(self) -> Dict[str, Route]:
        """Initialize routes with expanded examples"""
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
                patterns=[], keywords=[], phrase_patterns=[]
            ),
            
            'agent_request': Route(
                name='agent_request',
                examples=[
                    'I need to speak to an agent',
                    'connect me with a representative',
                    'transfer me to a human',
                    'I want to talk to someone',
                    'can I speak with an agent',
                    'get me a real person',
                    'I need human assistance',
                    'connect me to support',
                    'speak to customer service',
                    'talk to a representative',
                    'escalate to agent',
                    'I need a human',
                    'I want human help',
                    'live agent please',
                    'switch to human',
                    'operator please',
                    'get me an agent now',
                    'I want to talk to sales',
                    'I want a call from sales',
                    'I want to talk to the sales team',
                    'let me talk to sales now',
                ],
                patterns=[], keywords=[], phrase_patterns=[]
            ),
            
            'scheduler': Route(
                name='scheduler',
                examples=[
                    'I want to book an appointment',
                    'schedule a meeting',
                    'can I make an appointment',
                    'book a consultation',
                    'I need to schedule something',
                    'set up an appointment',
                    'arrange a meeting',
                    'I want to reserve a time slot',
                    'schedule an appointment for next week',
                    'book me for tomorrow',
                    'make a reservation',
                    'set up a meeting time',
                    'reserve a time',
                    'book a session',
                    'schedule a call',
                    'I want to schedule my meeting',
                    'I need to book a meeting',
                    'schedule my appointment',
                ],
                patterns=[], keywords=[], phrase_patterns=[]
            ),
            
            'conversation_close': Route(
                name='conversation_close',
                examples=[
                    'no', 'nope', 'no thanks', 'no thank you',
                    'nothing', 'nothing else', 'i dont need anything',
                    'i dont need any answer', 'i dont want anything',
                    'no need', 'im fine', 'i am fine', 'im good', 'i am good',
                    'bye', 'goodbye', 'see you', 'take care',
                    'i am done', 'im done', 'done', 'all done',
                    'thats all', 'that is all', 'nothing more',
                    'never mind', 'forget it', 'leave it',
                    'no more questions', 'no i am ok',
                    'not interested', 'not right now', 'maybe later',
                    'stop', 'end', 'close', 'exit',
                ],
                patterns=[], keywords=[], phrase_patterns=[]
            ),

            'abusive': Route(
                name='abusive',
                examples=[
                    'fuck you', 'you are an idiot', 'this is bullshit',
                    'what the hell', 'piss off', 'you are useless',
                    'go to hell', 'damn it', 'you suck', 'asshole',
                    'you are a moron', 'stupid bot', 'this is garbage',
                    'kill yourself', 'i hate you', 'you are worthless',
                    'go fuck yourself', 'eat shit', 'you are a joke',
                    'this is a waste of time', 'i will kill you'
                ],
                patterns=[], keywords=[], phrase_patterns=[]
            ),

            'normal_qa': Route(
                name='normal_qa',
                examples=[
                    'what are your business hours',
                    'how much does it cost',
                    'where are you located',
                    'what services do you offer',
                    'what is your return policy',
                    'how long does shipping take',
                    'I have a question about',
                    'tell me about your products',
                    'how does this work',
                    'explain the process',
                    'I want to know about handshaking',
                    'I want to understand TCP',
                    'what is data encryption',
                    'explain HTTP protocol',
                    'I want to learn about APIs',
                    'tell me about TCP/IP',
                    'what is the relationship between TCP and HTTP',
                    'I want to know about data',
                    'transport layer security',
                ],
                patterns=[], keywords=[], phrase_patterns=[]
            )
        }
    
    def _get_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        return mean_pooled[0].numpy()
    
    def _compute_route_embeddings(self) -> Dict[str, np.ndarray]:
        print("Computing BERT embeddings for routes...")
        embeddings = {}
        
        for route_name, route in self.routes.items():
            route_embeds = [self._get_embedding(ex) for ex in route.examples]
            embeddings[route_name] = np.mean(route_embeds, axis=0)
            print(f"  ✓ {route_name}: {len(route.examples)} examples")
        
        print()
        return embeddings
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def classify(self, query: str, return_scores: bool = False) -> Tuple[str, float, Dict]:
        if not query or not query.strip():
            return 'normal_qa', 0.0, {}

        # ── Early exact-match for conversation-close phrases ──
        _close_exact = {
            'no', 'nope', 'nah', 'bye', 'goodbye', 'done', 'nothing',
            'stop', 'end', 'exit', 'close', 'later', 'enough',
            'ok bye', 'okay bye', 'thanks bye', 'no thanks',
            'no thank you', 'im fine', 'i am fine', 'im good',
            'i am good', 'im done', 'i am done', 'all done',
            'thats all', 'that is all', 'thats it', 'that is it',
            'nothing else', 'no need', 'not needed', 'never mind',
            'nevermind', 'leave it', 'forget it', 'not interested',
            'not right now', 'maybe later', 'no i am ok', 'no its ok',
            'i dont need any answer', 'i do not need any answer',
            'i dont need anything', 'i do not need anything',
            'i dont want anything', 'i do not want anything',
        }
        q_norm = query.strip().lower()
        # Remove trailing punctuation for matching
        q_clean = re.sub(r'[^\w\s]', '', q_norm).strip()
        q_clean = re.sub(r'\s+', ' ', q_clean)
        if q_clean in _close_exact:
            if return_scores:
                return 'conversation_close', 0.98, {'method': 'bert_rule', 'reason': 'exact_close_match'}
            return 'conversation_close', 0.98, {}
        
        # ------------------------------------------------------------------
        # Heuristic routing for call vs. schedule semantics
        # - Immediate "talk/call now" b agent_request
        # - Explicit booking/scheduling or future-time calls b scheduler
        # ------------------------------------------------------------------
        q_lower = query.lower()

        # Include common misspellings ("shedule", "tommorow") and
        # generic booking words.
        schedule_keywords = [
            'schedule', 'scheduled', 'scheduling', 'shedule', 'booking',
            'book', 'appointment', 'reserve', 'reservation', 'slot',
            'time slot', 'reschedule'
        ]
        call_keywords = ['call', 'phone', 'meeting', 'session']
        human_keywords = [
            'agent', 'human', 'representative', 'person', 'someone',
            'support', 'customer service', 'sales', 'sales team', 'salesperson'
        ]
        # Future-time indicators so phrases like "call tomorrow",
        # "call after next 2 days" route to scheduler.
        future_time_keywords = [
            'tomorrow', 'tommorow', 'next ', 'later', 'after ',
            'in ', 'day after tomorrow'
        ]

        has_schedule_word = any(k in q_lower for k in schedule_keywords)
        has_call = any(k in q_lower for k in call_keywords)
        has_human = any(k in q_lower for k in human_keywords)
        has_future_time = any(k in q_lower for k in future_time_keywords)

        # If user explicitly talks about scheduling/booking OR clearly
        # mentions a future time with a call/meeting b scheduler.
        if has_call and (has_schedule_word or has_future_time):
            if return_scores:
                return 'scheduler', 0.95, {
                    'method': 'bert_rule',
                    'reason': 'schedule_or_future_call_heuristic'
                }
            return 'scheduler', 0.95, {}

        # If user wants to talk/call a human (including sales) and
        # there is no scheduling / future-time language b agent_request.
        if has_call and has_human and not (has_schedule_word or has_future_time):
            if return_scores:
                return 'agent_request', 0.95, {
                    'method': 'bert_rule',
                    'reason': 'immediate_call_heuristic'
                }
            return 'agent_request', 0.95, {}

        # ------------------------------------------------------------------
        # Default: use BERT similarity across routes
        # ------------------------------------------------------------------
        query_embedding = self._get_embedding(query)
        
        similarities = {}
        for route_name, route_embedding in self.route_embeddings.items():
            sim = self._cosine_similarity(query_embedding, route_embedding)
            similarities[route_name] = sim
        
        best_route = max(similarities, key=similarities.get)
        confidence = similarities[best_route]

        # If BERT picks agent_request, require BOTH an agent/human
        # marker AND a communication verb to prevent "sales services"
        # from escalating.
        if best_route == 'agent_request':
            q_words = set(re.findall(r"\w+", query.lower()))
            agent_markers = {
                'agent', 'human', 'representative', 'operator', 'someone',
                'person', 'support', 'customer', 'service', 'sales', 'team'
            }
            comm_verbs = {
                'talk', 'speak', 'call', 'connect', 'contact',
                'reach', 'transfer', 'escalate'
            }
            has_marker = any(k in q_words for k in agent_markers)
            has_verb = any(v in q_words for v in comm_verbs)
            if not (has_marker and has_verb):
                best_route = 'normal_qa'
                confidence = similarities['normal_qa']

        # If BERT picks scheduler, require at least one explicit
        # scheduling/booking word.  Prevents "How can I place an order?"
        # from routing to scheduler due to semantic similarity.
        if best_route == 'scheduler':
            q_words = set(re.findall(r"\w+", query.lower()))
            schedule_markers = {
                'schedule', 'scheduling', 'scheduled', 'reschedule',
                'book', 'booking', 'appointment', 'reservation',
                'reserve', 'slot', 'arrange', 'calendar',
                'meeting',
            }
            if not any(k in q_words for k in schedule_markers):
                best_route = 'normal_qa'
                confidence = similarities['normal_qa']

        max_specific_score = max(
            similarities.get('greeting', 0.0),
            similarities.get('agent_request', 0.0),
            similarities.get('scheduler', 0.0),
            similarities.get('conversation_close', 0.0)
        )
        
        if confidence < 0.05 or max_specific_score < 0.05:
            best_route = 'normal_qa'
            confidence = similarities['normal_qa']
        elif best_route != 'normal_qa' and confidence < self.confidence_threshold:
            if similarities['normal_qa'] >= confidence * 0.9:
                best_route = 'normal_qa'
                confidence = similarities['normal_qa']
        
        if return_scores:
            return best_route, confidence, {'method': 'bert', 'scores': similarities}
        
        return best_route, confidence, {}


# ============================================================================
# hybrid_classifier.py - Combines TF-IDF + BERT for best accuracy
# ============================================================================

class HybridIntentClassifier:
    """
    Hybrid classifier combining TF-IDF (fast) + BERT (accurate)
    - Uses TF-IDF for confident classifications (fast path)
    - Uses BERT for ambiguous cases (accurate path)
    """
    
    def __init__(
        self,
        use_bert_threshold: float = 0.70,
        confidence_threshold: float = 0.60,
        enable_bert: bool = True
    ):
        """
        Initialize hybrid classifier
        
        Args:
            use_bert_threshold: Use BERT if TF-IDF confidence below this
            confidence_threshold: Minimum confidence for classification
            enable_bert: Enable BERT classifier (set False if not installed)
        """
        print("Initializing Hybrid Intent Classifier...")
        print("=" * 70)
        
        # Always load TF-IDF (fast classifier)
        print("Loading TF-IDF classifier...")
        self.tfidf_classifier = SemanticRouteClassifier(confidence_threshold)
        print("✓ TF-IDF classifier ready (fast path)\n")
        
        # Optionally load BERT
        self.enable_bert = enable_bert and BERT_AVAILABLE
        self.bert_classifier = None
        self.bert_loaded = False
        self.use_bert_threshold = use_bert_threshold
        
        if self.enable_bert:
            print("✓ BERT classifier will load on first ambiguous query (accurate path)")
        else:
            print("⚠️  BERT disabled - using TF-IDF only")
        
        print("=" * 70)
        print()
    
    def _ensure_bert_loaded(self):
        """Lazy load BERT classifier on first use"""
        if not self.bert_loaded and self.enable_bert:
            print("\n" + "=" * 70)
            print("Loading BERT for high-accuracy classification...")
            print("=" * 70)
            try:
                self.bert_classifier = BERTIntentClassifier()
                self.bert_loaded = True
                print("=" * 70)
                print("✓ BERT ready for accurate classification")
                print("=" * 70)
                print()
            except Exception as e:
                print(f"⚠️  BERT loading failed: {e}")
                print("   Continuing with TF-IDF only")
                self.enable_bert = False
    
    def classify(self, query: str, return_scores: bool = False) -> Tuple[str, float, Dict]:
        """
        Classify using hybrid approach
        
        Returns:
            Tuple of (route, confidence, details)
        """
        if not query or not query.strip():
            return 'normal_qa', 0.0, {}

        # ------------------------------------------------------------------
        # Early guard: conversation_close detection for short phrases
        # like "no", "bye", "i dont need any answer", etc.
        # ------------------------------------------------------------------
        _close_exact = {
            'no', 'nope', 'nah', 'bye', 'goodbye', 'done', 'nothing',
            'stop', 'end', 'exit', 'close', 'later', 'enough',
            'ok bye', 'okay bye', 'thanks bye', 'no thanks',
            'no thank you', 'im fine', 'i am fine', 'im good',
            'i am good', 'im done', 'i am done', 'all done',
            'thats all', 'that is all', 'thats it', 'that is it',
            'nothing else', 'no need', 'not needed', 'never mind',
            'nevermind', 'leave it', 'forget it', 'not interested',
            'not right now', 'maybe later', 'no i am ok', 'no its ok',
            'i dont need any answer', 'i do not need any answer',
            'i dont need anything', 'i do not need anything',
            'i dont want anything', 'i do not want anything',
        }
        q_clean = re.sub(r'[^\w\s]', '', query.strip().lower()).strip()
        q_clean = re.sub(r'\s+', ' ', q_clean)
        if q_clean in _close_exact:
            if return_scores:
                return 'conversation_close', 0.98, {
                    'method': 'hybrid_rule',
                    'reason': 'exact_close_match'
                }
            return 'conversation_close', 0.98, {}

        # ------------------------------------------------------------------
        # Early guard: user explicitly says they want to talk to "you" or
        # "bot" but does NOT mention any human/agent/support words.
        # In this case, keep them with the bot (normal_qa) rather than
        # escalating to a human agent.
        # ------------------------------------------------------------------
        q_lower = query.lower()
        talk_markers = ['talk', 'speak', 'chat']
        bot_targets = [' you', 'you ', ' bot', 'bot ']
        human_markers = [
            'agent', 'human', 'representative', 'operator', 'someone',
            'person', 'support', 'customer service', 'customer-care',
            'customer care'
        ]

        mentions_talk = any(t in q_lower for t in talk_markers)
        mentions_bot_target = any(b in q_lower for b in bot_targets)
        mentions_human = any(h in q_lower for h in human_markers)

        if mentions_talk and mentions_bot_target and not mentions_human:
            if return_scores:
                return 'normal_qa', 0.95, {
                    'method': 'hybrid_rule',
                    'reason': 'talk_to_bot_or_you_guard'
                }
            return 'normal_qa', 0.95, {}

        # Strong escalation: communication verb + human/agent target
        # Only fire agent_request when user explicitly wants to
        # talk/call/connect to a person/agent/sales etc.
        comm_verbs = ['talk', 'speak', 'call', 'connect', 'contact', 'reach', 'transfer']
        escalation_targets = [
            'agent', 'human', 'representative', 'person', 'someone',
            'somebody', 'support', 'team', 'staff', 'sales', 'sales team',
            'salesperson', 'advisor', 'consultant', 'expert'
        ]
        direct_phrases = [
            'call me', 'call us', 'give me a call', 'ring me',
            'connect me', 'connect us'
        ]
        schedule_words = ['schedule', 'book', 'booking', 'appointment', 'slot']

        has_comm_verb = any(v in q_lower for v in comm_verbs)
        has_escalation_target = any(t in q_lower for t in escalation_targets)
        has_direct_phrase = any(p in q_lower for p in direct_phrases)
        has_schedule = any(s in q_lower for s in schedule_words)

        if (has_direct_phrase or (has_comm_verb and has_escalation_target)) and not has_schedule:
            if return_scores:
                return 'agent_request', 0.97, {
                    'method': 'hybrid_rule',
                    'reason': 'explicit_contact_phrase'
                }
            return 'agent_request', 0.97, {}

        # Step 1: Try TF-IDF first (fast path)
        route, confidence, tfidf_details = self.tfidf_classifier.classify(query, return_scores=True)
        # breakpoint()
        # Step 2: If confidence high enough, return immediately
        if confidence >= self.use_bert_threshold or not self.enable_bert:
            if return_scores:
                return route, confidence, {
                    'method': 'tfidf_fast',
                    'tfidf_confidence': confidence,
                    'details': tfidf_details
                }
            return route, confidence, {}
        
        # Step 3: Use BERT for ambiguous cases (accurate path)
        self._ensure_bert_loaded()
        
        if self.bert_classifier:
            bert_route, bert_conf, bert_details = self.bert_classifier.classify(query, return_scores=True)
            
            if return_scores:
                return bert_route, bert_conf, {
                    'method': 'bert_accurate',
                    'tfidf_route': route,
                    'tfidf_confidence': confidence,
                    'bert_route': bert_route,
                    'bert_confidence': bert_conf,
                    'bert_details': bert_details
                }
            
            return bert_route, bert_conf, {}
        
        # Fallback to TF-IDF if BERT failed
        if return_scores:
            return route, confidence, {
                'method': 'tfidf_fallback',
                'details': tfidf_details
            }
        return route, confidence, {}


# ============================================================================
# Integration with your existing code
# ============================================================================

# Singleton instance for reuse
_hybrid_classifier_instance = None

def get_hybrid_classifier():
    """Get cached hybrid classifier instance"""
    global _hybrid_classifier_instance
    if _hybrid_classifier_instance is None:
        _hybrid_classifier_instance = HybridIntentClassifier(
            use_bert_threshold=0.70,
            confidence_threshold=0.60,
            enable_bert=BERT_AVAILABLE
        )
    return _hybrid_classifier_instance


# ============================================================================
# Testing & Examples
# ============================================================================

if __name__ == "__main__":
    import time
    
    print("\n" + "=" * 90)
    print("HYBRID INTENT CLASSIFIER - COMPREHENSIVE TEST")
    print("=" * 90)
    print()
    
    # Test queries with expected results
    test_cases = [
        # Greetings
        ("Hello!", "greeting"),
        ("Good morning", "greeting"),
        ("Hey there", "greeting"),
        
        # Agent requests
        ("I need to speak with an agent", "agent_request"),
        ("Connect me to support", "agent_request"),
        ("Get me a human", "agent_request"),
        ("Transfer me to customer service", "agent_request"),
        
        # Scheduler
        ("I want to schedule my meeting", "scheduler"),
        ("Book an appointment for tomorrow", "scheduler"),
        ("I need to book a meeting", "scheduler"),
        ("Schedule me in", "scheduler"),
        ("Can I make an appointment", "scheduler"),
        
        # Normal QA
        ("What are your business hours?", "normal_qa"),
        ("How much does it cost?", "normal_qa"),
        ("Where are you located?", "normal_qa"),
        
        # Edge cases that were failing
        ("I want to know about handshaking", "normal_qa"),
        ("I want to know about data", "normal_qa"),
        ("I want to understand about TCP", "normal_qa"),
        ("What is the relationship between TCP and HTTP", "normal_qa"),
        ("Transport Layer Security", "normal_qa"),
        ("Tell me about HTTP protocol", "normal_qa"),
    ]
    
    # Initialize hybrid classifier
    classifier = get_hybrid_classifier()
    
    print("\n" + "=" * 90)
    print("CLASSIFICATION RESULTS")
    print("=" * 90)
    print()
    
    correct = 0
    total = len(test_cases)
    results_by_method = {'tfidf_fast': 0, 'bert_accurate': 0, 'tfidf_fallback': 0}
    
    for query, expected in test_cases:
        start = time.time()
        route, confidence, details = classifier.classify(query, return_scores=True)
        elapsed = (time.time() - start) * 1000
        
        is_correct = route == expected
        correct += is_correct
        
        method = details.get('method', 'unknown')
        results_by_method[method] = results_by_method.get(method, 0) + 1
        
        status = "✅" if is_correct else "❌"
        
        print(f"{status} {query:50} → {route:15}")
        print(f"   Expected: {expected:15} | Confidence: {confidence:.2%} | Method: {method:15} | Time: {elapsed:5.1f}ms")
        
        if not is_correct:
            print(f"   ⚠️  MISMATCH!")
        print()
    
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"Accuracy: {correct}/{total} = {(correct/total)*100:.1f}%")
    print(f"\nClassification Method Distribution:")
    for method, count in results_by_method.items():
        if count > 0:
            print(f"  • {method:20}: {count:2} queries ({(count/total)*100:.1f}%)")
    print()
    
    # Performance test
    print("=" * 90)
    print("PERFORMANCE TEST (100 queries)")
    print("=" * 90)
    
    perf_queries = [
        "Hello",
        "I need an agent",
        "Schedule a meeting",
        "What are your hours?",
        "I want to know about TCP"
    ] * 20  # 100 queries
    
    start_time = time.time()
    for q in perf_queries:
        classifier.classify(q)
    elapsed_time = time.time() - start_time
    
    avg_time = (elapsed_time / len(perf_queries)) * 1000
    throughput = len(perf_queries) / elapsed_time
    
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Average time per query: {avg_time:.2f}ms")
    print(f"Throughput: {throughput:.0f} queries/second")
    print()
    
    print("=" * 90)
    print("✅ TEST COMPLETE")
    print("=" * 90)