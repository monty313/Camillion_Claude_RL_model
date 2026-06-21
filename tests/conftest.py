# Ensures `from config...`/`from src...` work under pytest (repo root on path).
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
