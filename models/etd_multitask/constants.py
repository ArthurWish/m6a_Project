"""Shared constants for ETD multi-task m6A modeling."""

from __future__ import annotations

BASE_TO_ID = {
    "A": 0,
    "C": 1,
    "G": 2,
    "U": 3,
    "N": 4,
}
ID_TO_BASE = {value: key for key, value in BASE_TO_ID.items()}

PAD_TOKEN_ID = 5
MASK_TOKEN_ID = 6
APRIME_TOKEN_ID = 7
VOCAB_SIZE = 8

TASK_IDS = {
    "bind": 0,
    "mod": 1,
    "struct": 2,
    "mask": 3,
}

ROLE_IDS = {
    "none": 0,
    "writer": 1,
    "reader": 2,
    "eraser": 3,
}

COND_BASE_IDS = {
    "A": 0,
    "C": 1,
    "G": 2,
    "U": 3,
    "mask": 4,
}

ROLE_NAMES = ("writer", "reader", "eraser")
VALID_BASES = ("A", "C", "G", "U")
MOD_OUTPUT_BASES = ("A", "C", "U")
MOD_A_CHANNEL = 0
