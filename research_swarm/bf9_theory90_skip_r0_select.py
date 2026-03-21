"""
Theory 90: Skip selection in round 0 (only 1 option)

In round 0, idx starts at 0 (root). The "selection" is:
- idx = 1 + (val & 1)  [go left or right]

This is already the minimum: we NEED to compute which child.
The "1 option" framing is wrong - there ARE 2 children to choose from.

Theory N/A - round 0 selection cannot be skipped.
"""

import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problem import (
    SLOT_LIMITS, VLEN, N_CORES, Machine, Tree, Input,
    HASH_STAGES, build_mem_image, reference_kernel2, DebugInfo,
)
from collections import defaultdict


def test_theory():
    """Theory 90: Skip selection in round 0

    Round 0 operations in baseline:
    1. XOR val with tree[0]
    2. Hash (6 ops)
    3. bit0 = val & 1  (1 op)
    4. idx = 1 + bit0  (1 op)

    We cannot skip steps 3-4 because we NEED to determine
    which of 2 children to visit next.

    The framing "only 1 option" is incorrect - there are 2 children.
    Theory rejected.
    """
    return 1548


if __name__ == "__main__":
    cycles = test_theory()
    print(f"Theory 90 (Skip R0 select): {cycles} cycles")
    print(f"Delta from baseline (1548): {cycles - 1548}")
    print("Note: Round 0 has 2 child options, cannot skip selection")
