"""multitask.py — project-root entry point for the autograder.

The autograder does:
    from multitask import MultiTaskPerceptionModel

This file re-exports that class from models/multitask.py.
"""

from models.multitask import MultiTaskPerceptionModel 

__all__ = ["MultiTaskPerceptionModel"]