"""Tests for the reference KB builder — each table loads and is queryable."""

from __future__ import annotations

import sqlite3

import pytest


class TestKnowledgeBase:
    def test_all_tables_exist(self, temp_kb_path):
        conn = sqlite3.connect(temp_kb_path)
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {r[0] for r in cur.fetchall()}
        expected = {
            "rxnorm_cache",
            "renal_dosing_rules",
            "beers_criteria",
            "qtc_drugs",
            "anticholinergic_burden",
            "pregnancy_categories",
            "serotonergic_drugs",
            "drug_interactions",
        }
        assert expected.issubset(tables)
        conn.close()

    def test_renal_rules_populated(self, temp_kb_path):
        conn = sqlite3.connect(temp_kb_path)
        n = conn.execute("SELECT COUNT(*) FROM renal_dosing_rules").fetchone()[0]
        assert n > 20
        # Metformin must be there — it's the canonical example
        row = conn.execute(
            "SELECT action, rationale FROM renal_dosing_rules "
            "WHERE rxnorm_ingredient='6809' AND egfr_threshold=30"
        ).fetchone()
        assert row is not None
        assert row[0] == "AVOID"
        conn.close()

    def test_beers_populated(self, temp_kb_path):
        conn = sqlite3.connect(temp_kb_path)
        n = conn.execute("SELECT COUNT(*) FROM beers_criteria").fetchone()[0]
        assert n > 20
        # Diphenhydramine (3498) must appear
        row = conn.execute(
            "SELECT pim_category FROM beers_criteria WHERE rxnorm_ingredient='3498'"
        ).fetchone()
        assert row is not None
        conn.close()

    def test_ddi_table_populated(self, temp_kb_path):
        conn = sqlite3.connect(temp_kb_path)
        n = conn.execute("SELECT COUNT(*) FROM drug_interactions").fetchone()[0]
        assert n > 30
        # Warfarin (11289) × amiodarone (703) — the flagship example.
        # After fix_rxcuis.py, all pairs are stored in (smaller, larger) order.
        row = conn.execute(
            "SELECT base_severity FROM drug_interactions "
            "WHERE rxnorm_a IN ('11289','703') AND rxnorm_b IN ('11289','703')"
        ).fetchone()
        assert row is not None
        assert row[0] == "major"
        conn.close()

    def test_rebuild_is_idempotent(self, temp_kb_path):
        """Running build() twice should produce the same row counts, not duplicates."""
        from argus.reference.build_kb import build

        conn = sqlite3.connect(temp_kb_path)
        before = {
            "renal_dosing_rules": conn.execute("SELECT COUNT(*) FROM renal_dosing_rules").fetchone()[0],
            "beers_criteria": conn.execute("SELECT COUNT(*) FROM beers_criteria").fetchone()[0],
        }
        conn.close()

        build(temp_kb_path)  # rebuild

        conn = sqlite3.connect(temp_kb_path)
        after = {
            "renal_dosing_rules": conn.execute("SELECT COUNT(*) FROM renal_dosing_rules").fetchone()[0],
            "beers_criteria": conn.execute("SELECT COUNT(*) FROM beers_criteria").fetchone()[0],
        }
        conn.close()

        assert before == after
