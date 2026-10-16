import os
import yaml
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path=None):
    if path is None:
        path = os.path.join(BASE, "config", "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
