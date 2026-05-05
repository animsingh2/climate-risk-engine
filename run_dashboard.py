"""
Launch script — always run this from the project root.
Usage: streamlit run run_dashboard.py
"""
import sys
from pathlib import Path

# Add project root to path so 'src' is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.dashboard import *