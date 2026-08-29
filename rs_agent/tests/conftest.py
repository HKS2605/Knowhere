"""
conftest.py — pytest configuration for M2 test suite
Adds the project root to sys.path so pytest can import
`fusion.*` and `backbone.*` regardless of which directory
pytest is invoked from.
"""
import sys
from pathlib import Path

# Project root = one level up from this file (tests/)
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))