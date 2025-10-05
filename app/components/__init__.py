# ============= app/components/__init__.py =============
COMPONENTS_INIT = """
'''UI components package for Streamlit interface.'''

from .chat_interface import ChatInterface
from .report_generator import ReportGenerator

__all__ = ['ChatInterface', 'ReportGenerator']
"""