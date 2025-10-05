# ============= src/aggregation/__init__.py =============
'''Aggregation package for synthesizing multiple review sessions.'''

from .aggregator import PerformanceReviewAggregator
from .prompts import (
    MAP_ANALYSIS_TEMPLATE,
    REDUCE_CONSOLIDATION_TEMPLATE,
    TRIANGULATION_TEMPLATE,
    TREND_ANALYSIS_TEMPLATE,
    SMART_RECOMMENDATIONS_TEMPLATE
)

__all__ = [
    'PerformanceReviewAggregator',
    'MAP_ANALYSIS_TEMPLATE',
    'REDUCE_CONSOLIDATION_TEMPLATE',
    'TRIANGULATION_TEMPLATE',
    'TREND_ANALYSIS_TEMPLATE',
    'SMART_RECOMMENDATIONS_TEMPLATE'
]