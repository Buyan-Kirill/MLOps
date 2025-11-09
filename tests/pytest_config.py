import os
import sys


def pytest_configure(config):
    os.makedirs("logs", exist_ok=True)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
