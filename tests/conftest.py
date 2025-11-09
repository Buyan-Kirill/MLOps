#!/usr/bin/env python3
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, os.path.join(str(project_root), "scripts"))

os.makedirs("logs", exist_ok=True)
