"""Flush the RxNav cache table.

Run after fixing RxNav-related code to evict any polluted entries that were
written under the buggy version.

Usage:
    python scripts/flush_rxnorm_cache.py
"""

from __future__ import annotations

import sqlite3
import sys

from argus.config import get_settings


def main() -> int:
    path = str(get_settings().reference_kb_path)
    with sqlite3.connect(path) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM rxnorm_cache")
        before = cur.fetchone()[0]
        conn.execute("DELETE FROM rxnorm_cache")
        conn.commit()
    print(f"Flushed {before} cached entries from {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
