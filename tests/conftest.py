"""pytest fixtures and import path setup."""
import sys
from pathlib import Path

# Make canary/ importable from tests/
CANARY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CANARY_DIR))
