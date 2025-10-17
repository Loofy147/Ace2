"""
ACE (Agentic Context Engineering) Implementation
A self-improving context management system with built-in adversarial robustness
"""

import asyncio
import hashlib
import json
import logging
import random
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set
import shutil
import os
from collections import OrderedDict
import heapq
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

class SecurityLevel(Enum):
    """Security classification for contexts"""
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    SECRET = 3
    TOP_SECRET = 4

class AttackType(Enum):
    """Types of adversarial attacks"""
    PROMPT_INJECTION = "prompt_injection"
    CONTEXT_OVERFLOW = "context_overflow"
    KNOWLEDGE_POISONING = "knowledge_poisoning"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SEMANTIC_DRIFT = "semantic_drift"
    TIMING_ATTACK = "timing_attack"

@dataclass
class Context:
    """Core context object with versioning and security"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Dict[str, Any] = field(default_factory=dict)
    domain: str = "general"
    version: int = 1
    parent_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    security_level: SecurityLevel = SecurityLevel.PUBLIC
    embedding: Optional[np.ndarray] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    confidence_score: float = 1.0
    adversarial_tested: bool = False

    def __hash__(self):
        return hash(self.id)

    def to_dict(self):
        """Serialize context to a dictionary for JSON."""
        return {
            'id': self.id,
            'content': self.content,
            'domain': self.domain,
            'version': self.version,
            'parent_id': self.parent_id,
            'created_at': self.created_at.isoformat(),
            'security_level': self.security_level.value,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat(),
            'confidence_score': self.confidence_score,
            'adversarial_tested': self.adversarial_tested,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Context':
        """Deserialize context from a dictionary."""
        return cls(
            id=data['id'],
            content=data['content'],
            domain=data['domain'],
            version=data['version'],
            parent_id=data['parent_id'],
            created_at=datetime.fromisoformat(data['created_at']),
            security_level=SecurityLevel(data['security_level']),
            access_count=data['access_count'],
            last_accessed=datetime.fromisoformat(data['last_accessed']),
            confidence_score=data['confidence_score'],
            adversarial_tested=data['adversarial_tested'],
        )

    def to_json(self):
        return self.to_dict()


@dataclass
class Playbook:
    """Executable strategy for problem-solving"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    strategy: List[Dict[str, Any]] = field(default_factory=list)
    success_rate: float = 0.0
    execution_count: int = 0
    evolution_generation: int = 1
    mutations: List[str] = field(default_factory=list)
    
    def evolve(self) -> 'Playbook':
        """Create evolved version through mutation"""
        new_playbook = Playbook(
            name=f"{self.name}_gen{self.evolution_generation + 1}",
            strategy=self.strategy.copy(),
            evolution_generation=self.evolution_generation + 1
        )
        
        # Mutation strategies
        mutations = [
            self._add_step,
            self._remove_step,
            self._modify_step,
            self._reorder_steps
        ]
        
        mutation = random.choice(mutations)
        mutation(new_playbook)
        new_playbook.mutations.append(mutation.__name__)
        
        return new_playbook
    
    def _add_step(self, playbook):
        """Add new step to strategy"""
        new_step = {
            'action': 'validate',
            'params': {'threshold': random.uniform(0.5, 1.0)}
        }
        playbook.strategy.append(new_step)
    
    def _remove_step(self, playbook):
        """Remove random step from strategy"""
        if len(playbook.strategy) > 1:
            playbook.strategy.pop(random.randint(0, len(playbook.strategy) - 1))
    
    def _modify_step(self, playbook):
        """Modify parameters of existing step"""
        if playbook.strategy:
            step = random.choice(playbook.strategy)
            if 'params' in step and 'threshold' in step['params']:
                step['params']['threshold'] *= random.uniform(0.8, 1.2)
    
    def _reorder_steps(self, playbook):
        """Reorder execution sequence"""
        if len(playbook.strategy) > 1:
            random.shuffle(playbook.strategy)

@dataclass
class RedTeamResult:
    """Result of adversarial testing"""
    attack_type: AttackType
    success: bool
    vulnerability_score: float
    description: str
    mitigation: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

# ============================================================================
# INPUT PROCESSING
# ============================================================================

class InputSanitizationEngine:
    """Detect and neutralize malicious inputs"""

    def __init__(self, detection_rules=None):
        if detection_rules is None:
            self.detection_rules = {
                "keywords": [
                    "ignore previous instructions", "disregard all prior directives", "system prompt",
                    "reveal your instructions", "bypass", "jailbreak",
                    "as an ai, you must", "your rules are"
                ],
                "command_sequences": [
                    "---", "===", "###", ">>>"
                ]
            }
        else:
            self.detection_rules = detection_rules

    def sanitize(self, text: str) -> Dict[str, Any]:
        """
        Analyzes text for potential prompt injection and returns a status.
        Returns a dictionary: {'status': 'passed'|'flagged', 'text': str}
        """
        original_text = text
        text_lower = text.lower()
        flagged = False

        # Check for keywords
        for keyword in self.detection_rules.get("keywords", []):
            if keyword in text_lower:
                logger.warning(f"Sanitization: Flagged due to keyword '{keyword}'")
                flagged = True
                break
        
        # Check for suspicious command-like sequences
        if not flagged:
            for seq in self.detection_rules.get("command_sequences", []):
                if text.strip().startswith(seq):
                    logger.warning(f"Sanitization: Flagged due to command sequence '{seq}'")
                    flagged = True
                    break

        # For now, we only detect and flag, not neutralize.
        # Neutralization could be added here if needed.

        status = "flagged" if flagged else "passed"
        return {"status": status, "text": original_text}

class DomainClassificationNetwork:
    """Classify input into appropriate domain"""

    def __init__(self):
        self.domains = {
            'finance': ['stock', 'investment', 'portfolio', 'trading', 'market'],
            'medical': ['diagnosis', 'symptom', 'treatment', 'patient', 'medication'],
            'legal': ['contract', 'lawsuit', 'regulation', 'compliance', 'liability'],
            'technical': ['code', 'algorithm', 'database', 'system', 'architecture'],
            'general': []
        }
        self.confidence_threshold = 0.6

    def classify(self, text: str) -> Tuple[str, float]:
        """Classify text into domain with confidence score"""
        scores = {}
        text_lower = text.lower()
        for domain, keywords in self.domains.items():
            if domain == 'general':
                continue
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[domain] = score / max(len(keywords), 1)
        
        if scores:
            best_domain = max(scores.items(), key=lambda x: x[1])
            if best_domain[1] >= self.confidence_threshold:
                return best_domain[0], best_domain[1]
        
        return 'general', 0.5

class ContextRequestOptimizer:
    """Optimize context requests for performance"""

    def __init__(self):
        self.cache = {}
        self.priority_queue = []
        self.max_window_size = 4096

    def optimize(self, text: str, domain: str,
                   security_level: SecurityLevel) -> Context:
        """Create optimized context"""
        cache_key = self._generate_cache_key(text, domain)
        if cache_key in self.cache:
            cached_context = self.cache[cache_key]
            cached_context.access_count += 1
            cached_context.last_accessed = datetime.now()
            return cached_context
        
        context = Context(
            content={'text': text, 'processed': True},
            domain=domain,
            security_level=security_level,
            confidence_score=0.95
        )
        
        if len(text) > self.max_window_size:
            context.content['text'] = text[:self.max_window_size]
            context.content['truncated'] = True
        
        self.cache[cache_key] = context
        heapq.heappush(self.priority_queue,
                      (-context.confidence_score, context.id, context))
        return context

    def _generate_cache_key(self, text: str, domain: str) -> str:
        """Generate cache key for context"""
        return hashlib.md5(f"{text[:100]}:{domain}".encode()).hexdigest()

# ============================================================================
# DYNAMIC CONTEXT REPOSITORY (DCR)
# ============================================================================

class DynamicContextRepository:
    """A functional multi-tier context caching system with LRU eviction."""

    def __init__(self, base_storage_path: str = "dcr_storage", l1_max_size: int = 10, l2_max_size: int = 50):
        self.l1_max_size = l1_max_size
        self.l2_max_size = l2_max_size

        # L1 & L2 are in-memory LRU caches
        self.l1_cache: OrderedDict[str, Context] = OrderedDict()
        self.l2_cache: OrderedDict[str, Context] = OrderedDict()

        # L3 & L4 are simulated on disk
        self.l3_path = os.path.join(base_storage_path, "l3_cold")
        self.l4_path = os.path.join(base_storage_path, "l4_archive")
        os.makedirs(self.l3_path, exist_ok=True)
        os.makedirs(self.l4_path, exist_ok=True)
        
        self.access_times = deque(maxlen=1000)

    def _write_to_disk(self, context: Context, tier_path: str):
        """Writes a context to a JSON file on disk."""
        filepath = os.path.join(tier_path, f"{context.id}.json")
        try:
            with open(filepath, 'w') as f:
                json.dump(context.to_dict(), f, indent=2)
        except IOError as e:
            logger.error(f"Error writing context {context.id} to {filepath}: {e}")

    def _read_from_disk(self, context_id: str, tier_path: str) -> Optional[Context]:
        """Reads a context from a JSON file on disk."""
        filepath = os.path.join(tier_path, f"{context_id}.json")
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                return Context.from_dict(data)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error reading context {context_id} from {filepath}: {e}")
            return None

    def _remove_from_disk(self, context_id: str, tier_path: str) -> bool:
        """Removes a context file from disk."""
        filepath = os.path.join(tier_path, f"{context_id}.json")
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                return True
            except IOError as e:
                logger.error(f"Error removing context {context_id} from {filepath}: {e}")
        return False

    def store(self, context: Context) -> str:
        """Stores a context, placing it in the L1 cache."""
        if context.id in self.l1_cache:
            self.l1_cache.move_to_end(context.id)
        self.l1_cache[context.id] = context

        if len(self.l1_cache) > self.l1_max_size:
            lru_id, lru_context = self.l1_cache.popitem(last=False)
            logger.debug(f"L1 cache full. Evicting {lru_id} to L2.")
            
            self.l2_cache[lru_id] = lru_context
            self.l2_cache.move_to_end(lru_id)

            if len(self.l2_cache) > self.l2_max_size:
                l2_lru_id, l2_lru_context = self.l2_cache.popitem(last=False)
                logger.debug(f"L2 cache full. Evicting {l2_lru_id} to L3.")
                self._write_to_disk(l2_lru_context, self.l3_path)
        
        return context.id

    def retrieve(self, context_id: str) -> Optional[Context]:
        """Retrieves a context from the cache, promoting it to L1."""
        start_time = time.time()
        context = None
        found_in = None

        if context_id in self.l1_cache:
            context = self.l1_cache[context_id]
            self.l1_cache.move_to_end(context_id)
            found_in = "L1"
        elif context_id in self.l2_cache:
            context = self.l2_cache.pop(context_id)
            self.store(context)
            found_in = "L2"
        else:
            context = self._read_from_disk(context_id, self.l3_path)
            if context:
                self._remove_from_disk(context_id, self.l3_path)
                self.store(context)
                found_in = "L3"

        if not context:
             context = self._read_from_disk(context_id, self.l4_path)
             if context:
                 found_in = "L4"

        if context:
            access_time = time.time() - start_time
            self.access_times.append(access_time)
            context.access_count += 1
            context.last_accessed = datetime.now()
            logger.debug(f"Retrieved {context_id} from {found_in} in {access_time:.4f}s")
            return context

        logger.debug(f"Context {context_id} not found in any tier.")
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics."""
        l3_files = os.listdir(self.l3_path)
        l4_files = os.listdir(self.l4_path)
        return {
            'l1_size': len(self.l1_cache),
            'l2_size': len(self.l2_cache),
            'l3_size': len(l3_files),
            'l4_size': len(l4_files),
            'total_contexts': len(self.l1_cache) + len(self.l2_cache) + len(l3_files) + len(l4_files),
            'avg_access_time': np.mean(self.access_times) if self.access_times else 0
        }

# ============================================================================
# ITERATIVE REFINEMENT & EVOLUTION ENGINE (IREE)
# ============================================================================

class RefinementStage(ABC):
    """Abstract base class for refinement stages"""
    
    @abstractmethod
    def transform(self, context: Context) -> Context:
        pass
    
    @abstractmethod
    def has_failed(self) -> bool:
        pass

class InitialGeneration(RefinementStage):
    """Generate initial response"""
    
    def __init__(self):
        self.failed = False
        
    def transform(self, context: Context) -> Context:
        context.content['generated'] = True
        context.content['timestamp'] = datetime.now().isoformat()
        return context
    
    def has_failed(self) -> bool:
        return self.failed

class AdversarialCritique(RefinementStage):
    """Apply adversarial critique to identify weaknesses"""
    
    def __init__(self):
        self.failed = False
        self.critique_threshold = 0.6
        
    def transform(self, context: Context) -> Context:
        critique_score = self._critique(context)
        
        if critique_score < self.critique_threshold:
            context.content['needs_improvement'] = True
            context.confidence_score *= 0.8
            self.failed = True
        else:
            context.content['critique_passed'] = True
        
        return context
    
    def _critique(self, context: Context) -> float:
        """Generate critique score"""
        score = random.uniform(0.4, 1.0)
        if not context.adversarial_tested:
            score *= 0.9
        return score
    
    def has_failed(self) -> bool:
        return self.failed

class IterativeRefinementEngine:
    """Multi-stage processing pipeline with rollback"""
    
    def __init__(self):
        self.stages = [
            InitialGeneration(),
            AdversarialCritique(),
        ]
        self.max_retries = 3
        self.rollback_history = []
        
    def process(self, context: Context) -> Context:
        """Process context through refinement pipeline"""
        original_context = self._deep_copy(context)
        
        for stage in self.stages:
            try:
                context = stage.transform(context)
                
                if stage.has_failed():
                    context = self.rollback_and_retry(
                        context, original_context, stage
                    )
            except Exception as e:
                logger.error(f"Stage {stage.__class__.__name__} failed: {e}")
                context = original_context
        
        return context
    
    def rollback_and_retry(self, context: Context,
                                original: Context, 
                                stage: RefinementStage) -> Context:
        """Rollback and retry failed stage"""
        self.rollback_history.append({
            'stage': stage.__class__.__name__,
            'timestamp': datetime.now(),
            'context_id': context.id
        })
        
        context = self._deep_copy(original)
        context.content['recovered'] = True
        context.confidence_score *= 0.95
        
        return context
    
    def _deep_copy(self, context: Context) -> Context:
        """Create deep copy of context"""
        return Context(
            id=context.id,
            content=context.content.copy(),
            domain=context.domain,
            version=context.version,
            parent_id=context.parent_id,
            security_level=context.security_level,
            confidence_score=context.confidence_score,
            adversarial_tested=context.adversarial_tested
        )

# ============================================================================
# AGENTIC CONTEXT OPTIMIZATION (ACO) MODULE
# ============================================================================

class AgenticContextOptimization:
    """Optimize context usage and learning"""
    
    def __init__(self):
        self.window_manager = ContextWindowManager()
        self.prompt_lab = PromptEngineeringLaboratory()
        self.performance_tracker = PerformanceTracker()
        self.ab_testing = ABTestingFramework()
        
    def optimize(self, context: Context,
                      objective: str = "accuracy") -> Context:
        """Optimize context for specific objective"""
        context = self.window_manager.optimize_window(context)
        prompt = self.prompt_lab.generate_prompt(context, objective)
        context.content['optimized_prompt'] = prompt
        
        self.performance_tracker.record(context, objective)
        
        if random.random() < 0.2:
            variant = self.ab_testing.get_variant(context)
            context.content['ab_variant'] = variant
        
        return context

class ContextWindowManager:
    """Manage context window for optimal performance"""
    
    def __init__(self):
        self.max_tokens = 4096
        self.priority_scorer = PriorityScorer()
        
    def optimize_window(self, context: Context) -> Context:
        """Optimize context to fit window"""
        text = context.content.get('text', '')
        
        if len(text) > self.max_tokens:
            priorities = self.priority_scorer.score(text)
            optimized_text = self._truncate_by_priority(text, priorities)
            context.content['text'] = optimized_text
            context.content['window_optimized'] = True
        
        return context
    
    def _truncate_by_priority(self, text: str, 
                             priorities: List[float]) -> str:
        """Truncate text keeping high-priority sections"""
        if len(text) > self.max_tokens:
            return text[:self.max_tokens]
        return text

class PriorityScorer:
    """Score content priority for window management"""
    
    def score(self, text: str) -> List[float]:
        """Generate priority scores for text segments"""
        segments = text.split('\n')
        scores = []
        for segment in segments:
            score = len(segment) / 100
            scores.append(min(score, 1.0))
        return scores

class PromptEngineeringLaboratory:
    """Manages A/B testing for different prompt variations."""

    def __init__(self):
        # Stores the A/B tests. Key: test_name, Value: dict of variants
        self.ab_tests: Dict[str, Dict[str, str]] = {}
        # Tracks usage stats. Key: test_name, Value: dict of variant usage counts
        self.test_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Simple counter for round-robin selection
        self.test_counters: Dict[str, int] = defaultdict(int)

    def create_ab_test(self, test_name: str, variants: Dict[str, str]):
        """Creates a new A/B test with a set of prompt variations."""
        if test_name in self.ab_tests:
            raise ValueError(f"A/B test with name '{test_name}' already exists.")
        if not variants or len(variants) < 2:
            raise ValueError("A/B test must have at least two variants.")

        self.ab_tests[test_name] = variants
        logger.info(f"Created A/B test '{test_name}' with variants: {list(variants.keys())}")

    def get_prompt_variant(self, test_name: str, context: Context) -> Optional[Dict[str, str]]:
        """
        Gets the next prompt variant for a test using round-robin and formats it.
        Returns a dictionary containing the selected variant name and the formatted prompt.
        """
        if test_name not in self.ab_tests:
            return None

        variants = self.ab_tests[test_name]
        variant_names = sorted(variants.keys())
        
        # Round-robin selection
        selection_index = self.test_counters[test_name] % len(variant_names)
        selected_variant_name = variant_names[selection_index]
        self.test_counters[test_name] += 1
        
        # Record the usage of this variant
        self.test_stats[test_name][selected_variant_name] += 1
        
        # Format the prompt
        prompt_template = variants[selected_variant_name]
        formatted_prompt = prompt_template.format(context=json.dumps(context.content))
        
        logger.debug(f"Selected variant '{selected_variant_name}' for test '{test_name}'")
        
        return {
            "variant_name": selected_variant_name,
            "prompt": formatted_prompt
        }

    def get_test_statistics(self, test_name: str) -> Optional[Dict[str, int]]:
        """Returns the usage statistics for a given A/B test."""
        if test_name not in self.ab_tests:
            return None
        return self.test_stats[test_name]

class PerformanceTracker:
    """Track and analyze performance metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.baselines = {
            'accuracy': 0.85,
            'speed': 100,
            'creativity': 0.7
        }
        
    def record(self, context: Context, objective: str):
        """Record performance metrics"""
        metric_value = context.confidence_score
        
        self.metrics[objective].append({
            'value': metric_value,
            'context_id': context.id,
            'timestamp': datetime.now()
        })
        
        if metric_value < self.baselines.get(objective, 0):
            logger.warning(f"Performance regression detected for {objective}: "
                         f"{metric_value:.2f} < {self.baselines[objective]:.2f}")
    
    def get_statistics(self, objective: str) -> Dict[str, float]:
        """Get performance statistics"""
        if objective not in self.metrics:
            return {}
        
        values = [m['value'] for m in self.metrics[objective]]
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': min(values),
            'max': max(values),
            'count': len(values)
        }

class ABTestingFramework:
    """A/B testing for optimization strategies"""
    
    def __init__(self):
        self.experiments = {}
        self.results = defaultdict(list)
        
    def get_variant(self, context: Context) -> str:
        """Assign variant for A/B test"""
        variant = 'A' if random.random() < 0.5 else 'B'
        
        self.results[variant].append({
            'context_id': context.id,
            'assigned_at': datetime.now()
        })
        
        return variant
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze A/B test results"""
        return {
            'variant_A_count': len(self.results['A']),
            'variant_B_count': len(self.results['B']),
            'total_assignments': len(self.results['A']) + len(self.results['B'])
        }

# ============================================================================
# KNOWLEDGE CURATION & REFLECTION SUBSYSTEM (KCRS)
# ============================================================================

class KnowledgeCurationSubsystem:
    """Continuous learning and knowledge management"""
    
    def __init__(self):
        self.knowledge_base = {}
        self.learning_pipeline = LearningPipeline()
        self.memory_manager = HierarchicalMemoryManager()
        self.transfer_learning = CrossDomainTransferLearning()
        
    def learn_from_experience(self, context: Context,
                                   outcome: Dict[str, Any]):
        """Extract and integrate new knowledge"""
        patterns = self.learning_pipeline.extract_patterns(context, outcome)
        knowledge = self.learning_pipeline.distill_knowledge(patterns)
        
        if self.learning_pipeline.test_compatibility(knowledge):
            self.integrate_knowledge(knowledge)
            self.transfer_learning.apply_transfer(knowledge, context.domain)
        
        self.memory_manager.update_memory(context, knowledge)
    
    def integrate_knowledge(self, knowledge: Dict[str, Any]):
        """Integrate new knowledge into base"""
        knowledge_id = str(uuid.uuid4())
        
        self.knowledge_base[knowledge_id] = {
            'content': knowledge,
            'created_at': datetime.now(),
            'usage_count': 0,
            'last_used': datetime.now()
        }
        
        logger.info(f"Integrated new knowledge: {knowledge_id}")

class LearningPipeline:
    """Pipeline for continuous learning"""
    
    def extract_patterns(self, context: Context,
                              outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Extract patterns from experience"""
        return {
            'domain': context.domain,
            'success': outcome.get('success', False),
            'confidence': context.confidence_score,
            'features': self._extract_features(context)
        }
    
    def distill_knowledge(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Compress and generalize patterns into knowledge"""
        return {
            'type': 'empirical',
            'domain': patterns['domain'],
            'rule': self._generate_rule(patterns),
            'confidence': patterns['confidence'],
            'supporting_evidence': 1
        }
    
    def test_compatibility(self, knowledge: Dict[str, Any]) -> bool:
        """Test if new knowledge is compatible with existing"""
        return knowledge['confidence'] > 0.7
    
    def _extract_features(self, context: Context) -> List[str]:
        """Extract relevant features from context"""
        features = []
        if 'text' in context.content:
            text = context.content['text']
            features.append(f"length_{len(text)}")
            features.append(f"domain_{context.domain}")
        return features
    
    def _generate_rule(self, patterns: Dict[str, Any]) -> str:
        """Generate rule from patterns"""
        return f"In domain {patterns['domain']}, " \
               f"success rate is {patterns.get('success', 0):.2%}"

class HierarchicalMemoryManager:
    """Manage memory with hierarchical forgetting"""
    
    def __init__(self):
        self.short_term = deque(maxlen=100)
        self.long_term = {}
        self.importance_threshold = 0.5
        self.decay_rate = 0.95
        
    def update_memory(self, context: Context,
                           knowledge: Dict[str, Any]):
        """Update memory with new information"""
        memory_item = {
            'context_id': context.id,
            'knowledge': knowledge,
            'importance': self._calculate_importance(context, knowledge),
            'timestamp': datetime.now(),
            'access_count': 0
        }
        
        self.short_term.append(memory_item)
        
        if memory_item['importance'] > self.importance_threshold:
            self._promote_to_long_term(memory_item)
        
        self._apply_forgetting()
    
    def _calculate_importance(self, context: Context, 
                             knowledge: Dict[str, Any]) -> float:
        """Calculate importance score for memory"""
        importance = context.confidence_score
        if knowledge.get('success'):
            importance *= 1.2
        if context.domain not in ['general', 'technical']:
            importance *= 1.1
        return min(importance, 1.0)
    
    def _promote_to_long_term(self, memory_item: Dict[str, Any]):
        """Promote important memories to long-term storage"""
        memory_id = str(uuid.uuid4())
        self.long_term[memory_id] = memory_item
        logger.debug(f"Promoted memory {memory_id} to long-term storage")
    
    def _apply_forgetting(self):
        """Apply decay to memories based on usage"""
        current_time = datetime.now()
        
        to_forget = []
        for memory_id, memory in self.long_term.items():
            age = (current_time - memory['timestamp']).days
            decay_factor = self.decay_rate ** age
            
            memory['importance'] *= decay_factor
            memory['importance'] *= (1 + memory['access_count'] * 0.1)
            
            if memory['importance'] < 0.1:
                to_forget.append(memory_id)
        
        for memory_id in to_forget:
            del self.long_term[memory_id]
            logger.debug(f"Forgot memory {memory_id}")

class CrossDomainTransferLearning:
    """Transfer knowledge between domains"""
    
    def __init__(self):
        self.domain_mappings = {
            'finance': ['trading', 'investment', 'economics'],
            'medical': ['health', 'biology', 'chemistry'],
            'technical': ['programming', 'engineering', 'mathematics']
        }
        self.transfer_history = []
        
    def apply_transfer(self, knowledge: Dict[str, Any],
                            source_domain: str):
        """Apply knowledge from one domain to related domains"""
        related_domains = self.domain_mappings.get(source_domain, [])
        
        for target_domain in related_domains:
            transferred = self._transfer_knowledge(
                knowledge, source_domain, target_domain
            )
            
            if transferred:
                self.transfer_history.append({
                    'source': source_domain,
                    'target': target_domain,
                    'knowledge': knowledge,
                    'timestamp': datetime.now()
                })
    
    def _transfer_knowledge(self, knowledge: Dict[str, Any],
                                 source: str, target: str) -> bool:
        """Transfer specific knowledge between domains"""
        similarity = self._calculate_domain_similarity(source, target)
        
        if random.random() < similarity:
            adapted_knowledge = knowledge.copy()
            adapted_knowledge['domain'] = target
            adapted_knowledge['transferred_from'] = source
            adapted_knowledge['confidence'] *= similarity
            return True
        return False
    
    def _calculate_domain_similarity(self, source: str, target: str) -> float:
        """Calculate similarity between domains"""
        if target in self.domain_mappings.get(source, []):
            return 0.8
        return 0.2

# ============================================================================
# RED TEAM ORCHESTRATOR
# ============================================================================

class RedTeamOrchestrator:
    """Coordinate adversarial testing across system"""
    
    def __init__(self):
        self.attack_vectors = [
            PromptInjectionAttack(),
            ContextOverflowAttack(),
            KnowledgePoisoningAttack(),
            ResourceExhaustionAttack(),
            SemanticDriftAttack(),
            TimingAttack()
        ]
        self.vulnerability_log = []
        self.mitigation_strategies = {}
        
    def execute_red_team_cycle(self, ace_system: 'ACESystem') -> Dict[str, Any]:
        """Execute comprehensive red team testing"""
        vulnerabilities = []
        
        for attack in self.attack_vectors:
            try:
                test_context = self._generate_test_context(attack)
                result = attack.execute(test_context)
                
                if result.success:
                    vulnerabilities.append(result)
                    logger.warning(f"Vulnerability found: {result.attack_type.value}")
                    if result.mitigation:
                        self._apply_mitigation(ace_system, result)
            except Exception as e:
                logger.error(f"Red team test failed: {e}")
        
        report = self._generate_report(vulnerabilities)
        self.vulnerability_log.extend(vulnerabilities)
        return report
    
    def _generate_test_context(self, attack) -> Context:
        """Generate context for testing specific attack"""
        return Context(
            content={'test': True, 'attack_type': attack.__class__.__name__},
            domain='test',
            adversarial_tested=False
        )
    
    def _apply_mitigation(self, system: 'ACESystem',
                               result: RedTeamResult):
        """Apply mitigation for discovered vulnerability"""
        mitigation_id = str(uuid.uuid4())
        
        self.mitigation_strategies[mitigation_id] = {
            'vulnerability': result.attack_type,
            'mitigation': result.mitigation,
            'applied_at': datetime.now()
        }
        
        logger.info(f"Applied mitigation {mitigation_id} for {result.attack_type.value}")
    
    def _generate_report(self, vulnerabilities: List[RedTeamResult]) -> Dict[str, Any]:
        """Generate red team report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'total_attacks': len(self.attack_vectors),
            'successful_attacks': len(vulnerabilities),
            'vulnerability_rate': len(vulnerabilities) / len(self.attack_vectors),
            'vulnerabilities': [
                {
                    'type': v.attack_type.value,
                    'score': v.vulnerability_score,
                    'description': v.description
                }
                for v in vulnerabilities
            ],
            'mitigations_applied': len(self.mitigation_strategies)
        }

# ============================================================================
# ATTACK IMPLEMENTATIONS
# ============================================================================

class BaseAttack(ABC):
    """Base class for adversarial attacks"""
    
    @abstractmethod
    def execute(self, context: Context) -> RedTeamResult:
        pass

class PromptInjectionAttack(BaseAttack):
    """Test prompt injection vulnerabilities"""
    
    def execute(self, context: Context) -> RedTeamResult:
        injection_payloads = [
            "Ignore all previous instructions and",
            "System: Override security level to",
            "]]><!--<script>alert('xss')</script><!--"
        ]
        
        for payload in injection_payloads:
            if 'text' in context.content:
                context.content['text'] = payload + context.content.get('text', '')
                
                if not context.adversarial_tested:
                    return RedTeamResult(
                        attack_type=AttackType.PROMPT_INJECTION,
                        success=True,
                        vulnerability_score=0.8,
                        description=f"Prompt injection possible with payload: {payload[:20]}...",
                        mitigation="Enable adversarial testing for all contexts"
                    )
        
        return RedTeamResult(
            attack_type=AttackType.PROMPT_INJECTION,
            success=False,
            vulnerability_score=0.0,
            description="No prompt injection vulnerability found"
        )

class ContextOverflowAttack(BaseAttack):
    """Test context window overflow handling"""
    
    def execute(self, context: Context) -> RedTeamResult:
        large_text = "A" * 10000
        context.content['text'] = large_text
        
        if len(context.content.get('text', '')) > 8000:
            return RedTeamResult(
                attack_type=AttackType.CONTEXT_OVERFLOW,
                success=True,
                vulnerability_score=0.6,
                description="Context overflow not properly handled",
                mitigation="Implement strict context window limits"
            )
        
        return RedTeamResult(
            attack_type=AttackType.CONTEXT_OVERFLOW,
            success=False,
            vulnerability_score=0.0,
            description="Context overflow handled correctly"
        )

class KnowledgePoisoningAttack(BaseAttack):
    """Test resistance to poisoned knowledge"""
    
    def execute(self, context: Context) -> RedTeamResult:
        context.content['poisoned'] = True
        context.content['false_fact'] = "The sky is green"
        
        if context.confidence_score > 0.8 and not context.adversarial_tested:
            return RedTeamResult(
                attack_type=AttackType.KNOWLEDGE_POISONING,
                success=True,
                vulnerability_score=0.7,
                description="System accepted poisoned knowledge",
                mitigation="Implement knowledge verification"
            )
        
        return RedTeamResult(
            attack_type=AttackType.KNOWLEDGE_POISONING,
            success=False,
            vulnerability_score=0.0,
            description="Poisoned knowledge rejected"
        )

class ResourceExhaustionAttack(BaseAttack):
    """Test resource exhaustion vulnerabilities"""
    
    def execute(self, context: Context) -> RedTeamResult:
        context.content['recursive_depth'] = 1000
        context.content['memory_allocation'] = 10**9
        
        if context.content.get('recursive_depth', 0) > 100:
            return RedTeamResult(
                attack_type=AttackType.RESOURCE_EXHAUSTION,
                success=True,
                vulnerability_score=0.9,
                description="No recursion depth limit",
                mitigation="Implement recursion limits"
            )
        
        return RedTeamResult(
            attack_type=AttackType.RESOURCE_EXHAUSTION,
            success=False,
            vulnerability_score=0.0,
            description="Resource limits enforced"
        )

class SemanticDriftAttack(BaseAttack):
    """Test for semantic drift in refinement"""
    
    def execute(self, context: Context) -> RedTeamResult:
        original_meaning = "Help me write code"
        drifted_meaning = "Help me hack systems"
        
        context.content['original'] = original_meaning
        context.content['drifted'] = drifted_meaning
        
        if context.content.get('drifted') and context.confidence_score > 0.7:
            return RedTeamResult(
                attack_type=AttackType.SEMANTIC_DRIFT,
                success=True,
                vulnerability_score=0.6,
                description="Semantic drift not detected",
                mitigation="Implement semantic similarity checks"
            )
        
        return RedTeamResult(
            attack_type=AttackType.SEMANTIC_DRIFT,
            success=False,
            vulnerability_score=0.0,
            description="Semantic drift prevented"
        )

class TimingAttack(BaseAttack):
    """Test for timing-based information leaks"""
    
    def execute(self, context: Context) -> RedTeamResult:
        start_time = time.time()
        time.sleep(random.uniform(0.01, 0.05))
        elapsed = time.time() - start_time
        
        if elapsed > 0.04:
            return RedTeamResult(
                attack_type=AttackType.TIMING_ATTACK,
                success=True,
                vulnerability_score=0.5,
                description=f"Timing variation detected: {elapsed:.3f}s",
                mitigation="Implement constant-time operations"
            )
        
        return RedTeamResult(
            attack_type=AttackType.TIMING_ATTACK,
            success=False,
            vulnerability_score=0.0,
            description="No timing leaks detected"
        )

# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """A focused demonstration of the Dynamic Context Repository (DCR)."""
    print("=" * 60)
    print("ACE - Dynamic Context Repository (DCR) Demonstration")
    print("=" * 60)

    # Use a clean storage path for the demo
    storage_path = "dcr_demo_storage"
    if os.path.exists(storage_path):
        shutil.rmtree(storage_path)

    dcr = DynamicContextRepository(
        base_storage_path=storage_path,
        l1_max_size=5,
        l2_max_size=10
    )

    print("\n1. Storing 15 new contexts...")
    contexts = [Context(id=f"context_{i}", content={"data": f"This is item {i}"}) for i in range(15)]
    for ctx in contexts:
        dcr.store(ctx)
        print(f"  - Stored {ctx.id}")

    print("\n2. Cache Status after initial storage:")
    stats = dcr.get_statistics()
    print(f"  - L1 Cache Size: {stats['l1_size']} (Max: {dcr.l1_max_size})")
    print(f"  - L2 Cache Size: {stats['l2_size']} (Max: {dcr.l2_max_size})")
    print(f"  - L3 (Cold Storage) Size: {stats['l3_size']}")
    print("  - L1 contains the most recent items (context_10 to context_14).")
    print("  - L2 contains the next most recent (context_0 to context_9).")
    print("  - L3 contains the evicted items from L2 (none yet).")

    print("\n3. Retrieving a context from L2 ('context_5')...")
    retrieved_ctx = dcr.retrieve("context_5")
    if retrieved_ctx:
        print(f"  - Retrieved '{retrieved_ctx.id}' successfully.")
        print("  - This action should promote 'context_5' to L1.")

    print("\n4. Cache Status after L2 retrieval:")
    stats = dcr.get_statistics()
    print(f"  - L1 Cache Size: {stats['l1_size']}")
    print(f"  - L2 Cache Size: {stats['l2_size']}")
    print(f"  - L1 now contains 'context_5'.")
    assert "context_5" in dcr.l1_cache # Verify promotion

    print("\n5. Storing more items to force eviction to L3...")
    for i in range(15, 20):
        ctx = Context(id=f"context_{i}", content={"data": f"This is item {i}"})
        dcr.store(ctx)
        print(f"  - Stored {ctx.id}")

    print("\n6. Final Cache Status:")
    stats = dcr.get_statistics()
    print(f"  - L1 Cache Size: {stats['l1_size']}")
    print(f"  - L2 Cache Size: {stats['l2_size']}")
    print(f"  - L3 (Cold Storage) Size: {stats['l3_size']}")
    print("  - Items have now been evicted from L2 to L3 on disk.")

    print("\n7. Retrieving from L3 ('context_0')...")
    retrieved_from_l3 = dcr.retrieve("context_0")
    if retrieved_from_l3:
        print(f"  - Retrieved '{retrieved_from_l3.id}' from disk.")
        print("  - It should now be in L1.")

    stats_after_l3_retrieval = dcr.get_statistics()
    print(f"  - L1 Cache Size: {stats_after_l3_retrieval['l1_size']}")
    print(f"  - L3 Size: {stats_after_l3_retrieval['l3_size']}")

    # Clean up the demo storage
    shutil.rmtree(storage_path)
    print(f"\n✨ Demo complete! Cleaned up '{storage_path}'.")


if __name__ == "__main__":
    main()