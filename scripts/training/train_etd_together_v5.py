"""Legacy compatibility wrapper.

The maintained ETD-together entrypoint is:
    scripts/training/train_etd_together.py

The older v5 implementation lives under:
    scripts/training/legacy/train_etd_together_v5.py
"""

from scripts.training.legacy.train_etd_together_v5 import *  # noqa: F401,F403

