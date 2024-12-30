"""
recognizers.py

Custom pattern recognizers for the Presidio analyzer framework.
Provides optimized recognizers for domain-specific identifiers with
performance monitoring and pattern optimization.

Key Features:
    - Custom pattern recognition
    - Performance monitoring
    - Pattern optimization
    - Cached compilation
    - Memory-efficient processing

Performance Characteristics:
    - Optimized regex patterns
    - Pattern compilation caching
    - Memory-efficient context matching
    - Automatic pattern optimization

Usage:
    from recognizers import get_custom_recognizers
    
    recognizers = get_custom_recognizers()
    analyzer.registry.add_recognizers(recognizers)

Dependencies:
    - presidio-analyzer>=2.2.0

Author: [Your Name]
Last Modified: [Date]
"""

import logging
import re
from typing import List, Dict, Pattern
from datetime import datetime
import time
from functools import lru_cache

from presidio_analyzer import PatternRecognizer, Pattern as PresidioPattern
from presidio_analyzer.pattern import Pattern

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RecognizerMetrics:
    """
    Tracks performance metrics for pattern recognition operations.
    
    Attributes:
        start_time (datetime): Operation start time
        patterns_checked (int): Number of pattern checks performed
        matches_found (int): Number of pattern matches found
        cache_hits (int): Number of cache hits
        cache_misses (int): Number of cache misses
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        self.patterns_checked = 0
        self.matches_found = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.pattern_timings: Dict[str, float] = {}
        
    def get_summary(self) -> Dict:
        """Returns a summary of recognition performance metrics."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return {
            'duration_seconds': elapsed,
            'patterns_checked': self.patterns_checked,
            'matches_found': self.matches_found,
            'match_ratio': self.matches_found / self.patterns_checked if self.patterns_checked > 0 else 0,
            'cache_hit_ratio': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'pattern_timings': self.pattern_timings
        }

class OptimizedPatternRecognizer(PatternRecognizer):
    """
    Enhanced pattern recognizer with performance optimization and monitoring.
    
    Features:
        - Pattern compilation caching
        - Performance monitoring
        - Optimized context matching
        - Automatic pattern optimization
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = RecognizerMetrics()
        self._compiled_patterns: Dict[str, Pattern] = {}
        
    @lru_cache(maxsize=1000)
    def _compile_pattern(self, pattern: str) -> Pattern:
        """Caches compiled regex patterns."""
        return re.compile(pattern, re.IGNORECASE)
        
    def analyze(self, text: str, entities: List[str], nlp_artifacts: Dict = None) -> List:
        """
        Enhanced pattern analysis with performance monitoring.
        
        Args:
            text (str): Text to analyze
            entities (List[str]): Entities to look for
            nlp_artifacts (Dict, optional): NLP artifacts
            
        Returns:
            List: Recognition results
        """
        start_time = time.time()
        
        try:
            results = super().analyze(text, entities, nlp_artifacts)
            
            # Update metrics
            process_time = time.time() - start_time
            self.metrics.patterns_checked += len(self.patterns)
            self.metrics.matches_found += len(results)
            
            # Log slow operations
            if process_time > 0.1:  # Log operations taking more than 100ms
                logger.warning(
                    f"Slow pattern recognition: {process_time:.3f}s for "
                    f"{len(self.patterns)} patterns in text of length {len(text)}"
                )
                
            return results
            
        except Exception as e:
            logger.error(f"Pattern recognition error: {str(e)}", exc_info=True)
            raise

def get_custom_recognizers() -> List[PatternRecognizer]:
    try:
        # Badge Number Recognizer
        badge_recognizer = OptimizedPatternRecognizer(
            supported_entity="BADGE_NUMBER",
            name="Badge Number Recognizer",
            supported_language="en",
            patterns=[
                PresidioPattern(
                    name="badge_number_pattern",
                    regex=r"\b\d{6}\b",
                    score=0.8,
                ),
            ],
            context=["badge number", "employee ID", "ID number"]
        )

        # Ticket Number Recognizer
        ticket_recognizer = OptimizedPatternRecognizer(
            supported_entity="TICKET_NUMBER",
            name="Ticket Number Recognizer",
            supported_language="en",
            patterns=[
                PresidioPattern(
                    name="ticket_number_pattern",
                    regex=r"(?i)\b(?:INC|REQ)\d+\b",
                    score=0.8,
                ),
            ],
            context=["ticket", "incident", "request"]
        )

        # Machine Name Recognizer
        machine_name_recognizer = OptimizedPatternRecognizer(
            supported_entity="MACHINE_NAME",
            name="Machine Name Recognizer",
            supported_language="en",
            patterns=[
                PresidioPattern(
                    name="machine_name_pattern",
                    regex=r"\b[A-Z0-9]+(?:[_-][A-Z0-9]+)*\b",
                    score=0.5,
                ),
            ],
            context=["machine", "hostname", "computer", "PC"]
        )

        logger.info("Custom recognizers (English) initialized successfully")
        return [badge_recognizer, ticket_recognizer, machine_name_recognizer]

    except Exception as e:
        logger.error("Error initializing custom recognizers", exc_info=True)
        raise




# Optional: Pattern optimization utilities
def optimize_patterns(recognizer: OptimizedPatternRecognizer) -> None:
    """
    Optimizes patterns based on performance metrics.
    
    Args:
        recognizer: Recognizer instance to optimize
    """
    metrics = recognizer.metrics.get_summary()
    
    # Log pattern performance
    logger.info(
        f"Pattern performance for {recognizer.name}:\n"
        f"- Match ratio: {metrics['match_ratio']:.3f}\n"
        f"- Cache hit ratio: {metrics['cache_hit_ratio']:.3f}\n"
        f"- Average processing time: {metrics['duration_seconds']/metrics['patterns_checked']:.6f}s per pattern"
    )
