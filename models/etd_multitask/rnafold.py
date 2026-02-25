"""RNAfold helpers and dot-plot parsing."""

from __future__ import annotations

from pathlib import Path


def parse_dot_ps_ubox(dot_ps_path: str | Path) -> dict[tuple[int, int], float]:
    """Parse RNAfold dot plot (`*_dp.ps`) into sparse upper-triangle probabilities.

    Returns:
      dict[(i, j)] = p where i,j are 0-based and i < j.
    """
    dot_ps_path = Path(dot_ps_path)
    pairs: dict[tuple[int, int], float] = {}

    in_block = False
    with dot_ps_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not in_block:
                if stripped.startswith("%start of base pair probability data"):
                    in_block = True
                continue

            if not stripped:
                continue
            if stripped.startswith("showpage"):
                break
            if "ubox" not in stripped:
                continue

            parts = stripped.split()
            if len(parts) < 4:
                continue
            try:
                i = int(parts[0]) - 1
                j = int(parts[1]) - 1
                p_sqrt = float(parts[2])
            except ValueError:
                continue

            if i == j:
                continue
            if i > j:
                i, j = j, i

            p = p_sqrt * p_sqrt
            key = (i, j)
            if key in pairs:
                pairs[key] = max(pairs[key], p)
            else:
                pairs[key] = p

    return pairs
