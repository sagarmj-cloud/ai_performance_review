
"""
__init__.py files for all packages in the project.

These files make the directories importable as Python packages and
can optionally expose key classes and functions at the package level.
"""

# ============= config/__init__.py =============

'''Configuration package for the AI Performance Review System.'''
from .settings import settings

__all__ = ['settings']
