"""Chess Feature Explainability Audit Tool.

A comprehensive tool for auditing the explainability of chess feature sets
against Stockfish engine evaluations.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .audit import AuditResult, audit_feature_set
from .engine import SFConfig, sf_eval, sf_open, sf_top_moves
from .features import baseline_extract_features, load_feature_module

__all__ = [
    "AuditResult",
    "audit_feature_set",
    "baseline_extract_features",
    "load_feature_module",
    "SFConfig",
    "sf_eval",
    "sf_open",
    "sf_top_moves",
]
