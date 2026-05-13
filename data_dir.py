"""
Persistent data directory.
- On Railway: set DATA_DIR=/data environment variable + mount a volume at /data
- Local dev: falls back to the project directory
"""
import os

DATA_DIR = os.environ.get('DATA_DIR', os.path.dirname(os.path.abspath(__file__)))

def data_path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)
