"""Shared project configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
KNOWLEDGE_BASE_DIR = PROCESSED_DATA_DIR / "knowledge_base"

CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "wc2026_knowledge")

KAGGLE_DATASETS = {
    "wc2026_match_probability_baseline": "sarazahran1/wc2026-match-probability-baseline-dataset",
    "fifa_world_cup_2026_match_data": "areezvisram12/fifa-world-cup-2026-match-data-unofficial",
    "fifa_world_cup_team_dataset": "harrachimustapha/fifa-world-cup-team-dataset",
}

SUGGESTED_QUESTIONS = [
    "When are Mexico's group-stage games scheduled?",
    "Which teams are most likely to advance from the group stage?",
    "Simulate a likely path from group stage to the final.",
    "What team could be the surprise of this World Cup?",
    "What evidence did you retrieve for that answer?",
    "Which dataset fields look useful for match projections?",
    "Explain how the vector store was created.",
]
