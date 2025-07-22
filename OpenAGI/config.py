"""Configuration for the speculative planning system."""

from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Config:
    """Global configuration and metrics."""
    # Constants
    MAX_STEP: int = 5
    TRAIN_INTERVAL: int = 1
    WARMUP: int = 0
    
    # Flags
    ENABLE_TRAIN: bool = True
    ENABLE_PRED: bool = True
    USERINPUT: bool = False
    
    # Metrics
    MAX_CONCURRENT_CALLS: int = 0
    TOTAL_APPROXIMATION_CALLS: int = 0
    TOTAL_CORRECT_APPROXIMATION_CALLS: int = 0
    
    # Token tracking
    TOTAL_TOKEN_GENERATION: int = 0
    TOTAL_TOKEN_PROMPT: int = 0
    
    TARGET_NORMAL_PROMPT: Dict[int, int] = field(default_factory=dict)
    TARGET_NORMAL_GENERATION: Dict[int, int] = field(default_factory=dict)
    TARGET_SP_PROMPT: int = 0
    TARGET_SP_GENERATION: int = 0
    
    APPROX_SP_PROMPT: int = 0
    APPROX_SP_GENERATION: int = 0
    APPROX_NORMAL_PROMPT: Dict[int, int] = field(default_factory=dict)
    APPROX_NORMAL_GENERATION: Dict[int, int] = field(default_factory=dict)
    
    # Timing metrics
    TOTAL_SP_TIME: float = 0.0
    TARGET_NORMAL_TIME: Dict[int, float] = field(default_factory=dict)
    APPROX_NORMAL_TIME: Dict[int, float] = field(default_factory=dict)
    
    # Prediction metrics
    PREDICT_K: List[int] = field(default_factory=list)
    PREDICT_CORRECT: int = 0
    PREDICT_TOTAL: int = 0
    BUILD_TRAJ_TIMES: int = 0
    
    def reset_task_metrics(self):
        """Reset metrics for a new task."""
        self.MAX_CONCURRENT_CALLS = 0
        self.TOTAL_APPROXIMATION_CALLS = 0
        self.TOTAL_CORRECT_APPROXIMATION_CALLS = 0
        
        self.TOTAL_TOKEN_GENERATION = 0
        self.TOTAL_TOKEN_PROMPT = 0
        self.USERINPUT = False
        
        self.TARGET_NORMAL_PROMPT.clear()
        self.TARGET_NORMAL_GENERATION.clear()
        self.TARGET_SP_PROMPT = 0
        self.TARGET_SP_GENERATION = 0
        
        self.APPROX_SP_PROMPT = 0
        self.APPROX_SP_GENERATION = 0
        self.APPROX_NORMAL_PROMPT.clear()
        self.APPROX_NORMAL_GENERATION.clear()
        
        self.TOTAL_SP_TIME = 0.0
        self.TARGET_NORMAL_TIME.clear()
        self.APPROX_NORMAL_TIME.clear()
        
        self.PREDICT_K.clear()
        self.PREDICT_CORRECT = 0
        self.PREDICT_TOTAL = 0
        self.BUILD_TRAJ_TIMES = 0

    def initialize_from_args(self, args):
        """Initialize configuration from command line arguments."""
        self.TRAIN_INTERVAL = args.freq
        self.BUILD_TRAJ_TIMES = 0
        self.MAX_STEP = 5
        self.ENABLE_TRAIN = args.pred
        self.ENABLE_PRED = args.pred
        self.WARMUP = 0 