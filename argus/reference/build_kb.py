"""Reference KB builder.

Creates (or rebuilds) the SQLite database used by all tools at
`argus/reference/reference.sqlite`. Idempotent — safe to run repeatedly.

Tables:
    rxnorm_cache               (populated at runtime by argus.rxnorm)
    renal_dosing_rules         (seeded from data/renal_rules.csv)
    beers_criteria             (seeded from data/beers_2023.csv)
    qtc_drugs                  (seeded from data/qtc_drugs.csv)
    anticholinergic_burden     (seeded from data/anticholinergic_burden.csv)
    pregnancy_categories       (seeded from data/pregnancy_categories.csv)
    serotonergic_drugs         (seeded from data/serotonergic_drugs.csv)
    drug_interactions          (seeded from data/drug_interactions.csv)

All tables indexed on rxnorm_ingredient for sub-millisecond lookups.
"""

from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

from argus.config import get_settings
from argus.logging_setup import configure_logging, get_logger

log = get_logger(__name__)

DATA_DIR = Path(__file__).parent / "data"

SCHEMAS = {
    "rxnorm_cache": """
        CREATE TABLE IF NOT EXISTS rxnorm_cache (
            rxcui TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            fetched_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_rxnorm_cache_fetched ON rxnorm_cache(fetched_at);
    """,
    "renal_dosing_rules": """
        CREATE TABLE IF NOT EXISTS renal_dosing_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rxnorm_ingredient TEXT NOT NULL,
            ingredient_name TEXT,
            egfr_threshold REAL NOT NULL,
            action TEXT NOT NULL CHECK(action IN ('AVOID','REDUCE','MONITOR','NO_CHANGE')),
            adjusted_dose_pattern TEXT,
            rationale TEXT NOT NULL,
            source TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_renal_ingredient ON renal_dosing_rules(rxnorm_ingredient);
    """,
    "beers_criteria": """
        CREATE TABLE IF NOT EXISTS beers_criteria (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rxnorm_ingredient TEXT NOT NULL,
            ingredient_name TEXT,
            pim_category TEXT NOT NULL,
            rationale TEXT NOT NULL,
            alternative TEXT,
            severity TEXT CHECK(severity IN ('minor','moderate','major','critical'))
        );
        CREATE INDEX IF NOT EXISTS idx_beers_ingredient ON beers_criteria(rxnorm_ingredient);
    """,
    "qtc_drugs": """
        CREATE TABLE IF NOT EXISTS qtc_drugs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rxnorm_ingredient TEXT NOT NULL,
            ingredient_name TEXT,
            risk_category TEXT CHECK(risk_category IN ('known_risk','possible_risk','conditional_risk'))
        );
        CREATE INDEX IF NOT EXISTS idx_qtc_ingredient ON qtc_drugs(rxnorm_ingredient);
    """,
    "anticholinergic_burden": """
        CREATE TABLE IF NOT EXISTS anticholinergic_burden (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rxnorm_ingredient TEXT NOT NULL,
            ingredient_name TEXT,
            acb_score INTEGER NOT NULL CHECK(acb_score BETWEEN 1 AND 3)
        );
        CREATE INDEX IF NOT EXISTS idx_acb_ingredient ON anticholinergic_burden(rxnorm_ingredient);
    """,
    "pregnancy_categories": """
        CREATE TABLE IF NOT EXISTS pregnancy_categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rxnorm_ingredient TEXT NOT NULL,
            ingredient_name TEXT,
            category TEXT NOT NULL,
            pllr_summary TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_pregnancy_ingredient ON pregnancy_categories(rxnorm_ingredient);
    """,
    "serotonergic_drugs": """
        CREATE TABLE IF NOT EXISTS serotonergic_drugs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rxnorm_ingredient TEXT NOT NULL,
            ingredient_name TEXT,
            mechanism TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_serotonergic_ingredient ON serotonergic_drugs(rxnorm_ingredient);
    """,
    "drug_interactions": """
        CREATE TABLE IF NOT EXISTS drug_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rxnorm_a TEXT NOT NULL,
            name_a TEXT,
            rxnorm_b TEXT NOT NULL,
            name_b TEXT,
            base_severity TEXT NOT NULL CHECK(base_severity IN ('minor','moderate','major','critical')),
            mechanism TEXT NOT NULL,
            evidence_url TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_ddi_a ON drug_interactions(rxnorm_a);
        CREATE INDEX IF NOT EXISTS idx_ddi_b ON drug_interactions(rxnorm_b);
        CREATE INDEX IF NOT EXISTS idx_ddi_pair ON drug_interactions(rxnorm_a, rxnorm_b);
    """,
}

SEED_LOADERS = {
    "renal_dosing_rules": ("renal_rules.csv", [
        "rxnorm_ingredient",
        "ingredient_name",
        "egfr_threshold",
        "action",
        "adjusted_dose_pattern",
        "rationale",
        "source",
    ]),
    "beers_criteria": ("beers_2023.csv", [
        "rxnorm_ingredient",
        "ingredient_name",
        "pim_category",
        "rationale",
        "alternative",
        "severity",
    ]),
    "qtc_drugs": ("qtc_drugs.csv", [
        "rxnorm_ingredient",
        "ingredient_name",
        "risk_category",
    ]),
    "anticholinergic_burden": ("anticholinergic_burden.csv", [
        "rxnorm_ingredient",
        "ingredient_name",
        "acb_score",
    ]),
    "pregnancy_categories": ("pregnancy_categories.csv", [
        "rxnorm_ingredient",
        "ingredient_name",
        "category",
        "pllr_summary",
    ]),
    "serotonergic_drugs": ("serotonergic_drugs.csv", [
        "rxnorm_ingredient",
        "ingredient_name",
        "mechanism",
    ]),
    "drug_interactions": ("drug_interactions.csv", [
        "rxnorm_a",
        "name_a",
        "rxnorm_b",
        "name_b",
        "base_severity",
        "mechanism",
        "evidence_url",
    ]),
}


def build(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")

        # Create schemas
        for table, ddl in SCHEMAS.items():
            log.info("build_kb.create_table", table=table)
            conn.executescript(ddl)
        conn.commit()

        # Seed data
        for table, (filename, columns) in SEED_LOADERS.items():
            csv_path = DATA_DIR / filename
            if not csv_path.exists():
                log.warning("build_kb.seed_missing", table=table, file=str(csv_path))
                continue

            # Wipe existing seed rows — allows rerun to pick up CSV edits
            conn.execute(f"DELETE FROM {table}")

            with csv_path.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = []
                for r in reader:
                    row = tuple(_coerce(r.get(c), c) for c in columns)
                    rows.append(row)
            placeholders = ",".join(["?"] * len(columns))
            conn.executemany(
                f"INSERT INTO {table}({','.join(columns)}) VALUES ({placeholders})",
                rows,
            )
            log.info("build_kb.loaded", table=table, rows=len(rows))

        conn.commit()
    finally:
        conn.close()

    log.info("build_kb.done", path=str(db_path))


def _coerce(val, col: str):
    """Coerce CSV strings into the right types; treat '' as NULL."""
    if val is None or val == "":
        return None
    if col in ("egfr_threshold",):
        try:
            return float(val)
        except ValueError:
            return None
    if col in ("acb_score",):
        try:
            return int(val)
        except ValueError:
            return None
    return val


def main() -> int:
    configure_logging()
    settings = get_settings()
    db_path = Path(settings.reference_kb_path)
    build(db_path)
    print(f"✓ Reference KB built at {db_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
