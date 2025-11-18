"""Root conftest for all tests - setup sys.path for custom libraries."""

import sys
from pathlib import Path

# Add project root to sys.path for custom library imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
