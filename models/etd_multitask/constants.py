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

MOD_TOKEN_m6A  = 7   
MOD_TOKEN_m1A  = 8    
MOD_TOKEN_m5C  = 9   
MOD_TOKEN_pseu = 10   

VOCAB_SIZE = 11


MOD_TOKEN_IDS = {
    "m6A":  MOD_TOKEN_m6A,
    "m1A":  MOD_TOKEN_m1A,
    "m5C":  MOD_TOKEN_m5C,
    "pseu": MOD_TOKEN_pseu,
}



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




MOD_TYPE_IDS = {
    "none": 0,
    "m6A":  1,
    "m1A":  2,
    "m5C":  3,
    "pseu": 4,
}

MOD_TYPE_VOCAB = len(MOD_TYPE_IDS) 


MOD_BASE_MAP = {
    "m6A":  "A",
    "m1A":  "A",
    "m5C":  "C",
    "pseu": "U",
}

MOD_BASE_CHANNEL = {
    "m6A":  0,   # A 
    "m1A":  0,   # A 
    "m5C":  1,   # C 
    "pseu": 2,   # U
}

MOD_OUTPUT_BASES = ("A", "C", "U")

MOD_TYPE_NAMES = ("m6A", "m1A", "m5C", "pseu")


TASK_PROBS = {
    "bind": 0.1,
    "mod": 0.1,
    "struct": 0.0,
    "mask": 0.8,
}

