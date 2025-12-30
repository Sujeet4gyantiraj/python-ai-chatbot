class IntentRouter:
    """
    Advanced intent classification with multi-signal detection.
    Uses weighted pattern matching, sentiment analysis, and contextual rules.
    """
    
    # Strong signals (high confidence)
    GREETING_STRONG = {
        'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 
        'good evening', 'howdy', 'hiya', 'sup'
    }
    
    GREETING_WEAK = {
        'how are you', 'whats up', "what's up", 'how do you do', 'nice to meet',
        'thanks', 'thank you', 'thx', 'ty', 'appreciate', 'bye', 'goodbye', 
        'see you', 'take care', 'have a good', 'cheers'
    }
    
    AGENT_REQUEST_STRONG = {
        'speak to agent', 'talk to agent', 'human agent', 'real person', 
        'transfer me', 'connect me to', 'escalate', 'speak with someone',
        'talk to someone', 'customer service', 'support team', 'representative',
        'live agent', 'live support', 'speak to human', 'talk to human'
    }
    
    AGENT_REQUEST_WEAK = {
        'human', 'agent', 'person', 'someone', 'representative', 'rep',
        'manager', 'supervisor', 'team member', 'staff'
    }
    
    FRUSTRATION_SIGNALS = {
        'frustrated', 'angry', 'upset', 'disappointed', 'not helping',
        'useless', 'waste of time', 'ridiculous', 'terrible', 'awful',
        'horrible', 'pathetic', 'annoying', 'sick of', 'fed up', 'enough',
        'unacceptable', 'disgrace', 'joke', "can't believe", "won't help"
    }
    
    SCHEDULER_STRONG = {
        'book appointment', 'schedule meeting', 'book demo', 'schedule demo',
        'book a call', 'schedule call', 'set up meeting', 'arrange meeting',
        'book time', 'reserve time', 'schedule time', 'set up call',
        'arrange call', 'book consultation', 'schedule consultation'
    }
    
    SCHEDULER_WEAK = {
        'schedule', 'book', 'appointment', 'meeting', 'demo', 'call me',
        'callback', 'call back', 'reach out', 'contact me', 'get in touch',
        'available', 'when can', 'calendar', 'discuss', 'consultation',
        'convenient time', 'free time', 'talk soon'
    }
    
    # Question indicators (suggests QA intent)
    QUESTION_WORDS = {
        'what', 'when', 'where', 'why', 'how', 'who', 'which', 'whose',
        'is', 'are', 'can', 'could', 'would', 'should', 'do', 'does',
        'will', 'tell me', 'explain', 'describe', 'show me'
    }
    
    # Pricing/info keywords that override scheduler
    INFO_REQUEST_KEYWORDS = {
        'how much', 'cost', 'price', 'pricing', 'fee', 'charge', 'what is',
        'tell me about', 'information about', 'details about', 'explain',
        'describe', 'features', 'benefits', 'options', 'plans'
    }
    
    @classmethod
    def _contains_phrase(cls, message: str, patterns: set) -> tuple[bool, int]:
        """Check if message contains any patterns, return (found, match_count)"""
        matches = 0
        for pattern in patterns:
            if pattern in message:
                matches += 1
        return (matches > 0, matches)
    
    @classmethod
    def _is_question(cls, message: str) -> bool:
        """Detect if message is a question"""
        if '?' in message:
            return True
        words = message.split()
        if len(words) > 0 and words[0] in cls.QUESTION_WORDS:
            return True
        return False
    
    @classmethod
    def _calculate_weighted_score(cls, message: str, strong_patterns: set, 
                                   weak_patterns: set, additional_signals: set = None) -> float:
        """Calculate weighted confidence score for an intent"""
        score = 0.0
        
        strong_found, strong_count = cls._contains_phrase(message, strong_patterns)
        weak_found, weak_count = cls._contains_phrase(message, weak_patterns)
        
        if strong_found:
            score += 0.6 + (min(strong_count - 1, 2) * 0.15)  # 0.6 to 0.9
        
        if weak_found:
            score += 0.3 + (min(weak_count - 1, 2) * 0.1)  # 0.3 to 0.5
        
        if additional_signals:
            signal_found, signal_count = cls._contains_phrase(message, additional_signals)
            if signal_found:
                score += 0.2 + (min(signal_count - 1, 1) * 0.1)  # 0.2 to 0.3
        
        return min(score, 1.0)
    
    @classmethod
    def detect_intent(cls, user_message: str) -> str:
        """
        Detect intent using multi-signal analysis.
        
        Returns: 'greeting', 'agent_request', 'scheduler', or 'qa'
        """
        message_lower = user_message.lower().strip()
        message_len = len(message_lower.split())
        
        # Calculate scores for each intent
        scores = {}
        
        # GREETING SCORE
        scores['greeting'] = cls._calculate_weighted_score(
            message_lower, 
            cls.GREETING_STRONG, 
            cls.GREETING_WEAK
        )
        # Boost greeting score for very short messages (1-3 words)
        if message_len <= 3 and scores['greeting'] > 0:
            scores['greeting'] = min(scores['greeting'] + 0.3, 1.0)
        
        # AGENT REQUEST SCORE
        scores['agent_request'] = cls._calculate_weighted_score(
            message_lower,
            cls.AGENT_REQUEST_STRONG,
            cls.AGENT_REQUEST_WEAK,
            cls.FRUSTRATION_SIGNALS
        )
        # Big boost if frustration detected
        frustration_found, _ = cls._contains_phrase(message_lower, cls.FRUSTRATION_SIGNALS)
        if frustration_found:
            scores['agent_request'] = min(scores['agent_request'] + 0.3, 1.0)
        
        # SCHEDULER SCORE
        scores['scheduler'] = cls._calculate_weighted_score(
            message_lower,
            cls.SCHEDULER_STRONG,
            cls.SCHEDULER_WEAK
        )
        # Penalize scheduler if it's clearly an info request
        info_request, _ = cls._contains_phrase(message_lower, cls.INFO_REQUEST_KEYWORDS)
        if info_request:
            scores['scheduler'] *= 0.2  # Heavily penalize
        
        # QA SCORE (baseline + question boost)
        scores['qa'] = 0.4  # Base score for QA
        if cls._is_question(message_lower):
            scores['qa'] += 0.3
        
        # Find highest scoring intent
        max_intent = max(scores, key=scores.get)
        max_score = scores[max_intent]
        
        # Threshold check - if no strong signal, default to QA
        if max_score < 0.4 and max_intent != 'qa':
            return 'qa'
        
        return max_intent
    
    @classmethod
    def get_confidence_score(cls, user_message: str, detected_intent: str) -> float:
        """
        Calculate detailed confidence score for detected intent.
        """
        message_lower = user_message.lower().strip()
        
        if detected_intent == "greeting":
            score = cls._calculate_weighted_score(
                message_lower,
                cls.GREETING_STRONG,
                cls.GREETING_WEAK
            )
        elif detected_intent == "agent_request":
            score = cls._calculate_weighted_score(
                message_lower,
                cls.AGENT_REQUEST_STRONG,
                cls.AGENT_REQUEST_WEAK,
                cls.FRUSTRATION_SIGNALS
            )
        elif detected_intent == "scheduler":
            score = cls._calculate_weighted_score(
                message_lower,
                cls.SCHEDULER_STRONG,
                cls.SCHEDULER_WEAK
            )
            # Apply info request penalty
            info_request, _ = cls._contains_phrase(message_lower, cls.INFO_REQUEST_KEYWORDS)
            if info_request:
                score *= 0.3
        else:  # qa
            score = 0.5
            if cls._is_question(message_lower):
                score = 0.75
        
        return min(max(score, 0.1), 1.0)  # Clamp between 0.1 and 1.0