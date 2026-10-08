import sys
from pathlib import Path

# Make `import mtp` work before pyproject.toml / `pip install -e .` lands (OM-1).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
