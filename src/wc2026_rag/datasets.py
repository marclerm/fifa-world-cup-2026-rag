"""Download Kaggle datasets and convert tabular files into RAG-ready documents."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from wc2026_rag.config import KAGGLE_DATASETS, KNOWLEDGE_BASE_DIR, RAW_DATA_DIR


@dataclass(frozen=True)
class DatasetFile:
    dataset_key: str
    path: Path


def download_kaggle_datasets(force: bool = False) -> list[Path]:
    """Download the configured Kaggle datasets into data/raw.

    KaggleHub keeps its own cache, so this copies the latest cached files into this repo.
    """
    import kagglehub

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for key, handle in KAGGLE_DATASETS.items():
        target_dir = RAW_DATA_DIR / key
        if target_dir.exists() and force:
            shutil.rmtree(target_dir)
        if target_dir.exists() and any(target_dir.iterdir()):
            downloaded.append(target_dir)
            continue

        source_path = Path(kagglehub.dataset_download(handle))
        shutil.copytree(source_path, target_dir, dirs_exist_ok=True)
        downloaded.append(target_dir)

    return downloaded


def iter_dataset_files(raw_dir: Path = RAW_DATA_DIR) -> Iterable[DatasetFile]:
    """Yield supported data files found under data/raw."""
    extensions = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl", ".parquet"}
    for dataset_dir in sorted(p for p in raw_dir.glob("*") if p.is_dir()):
        for file_path in sorted(dataset_dir.rglob("*")):
            if file_path.suffix.lower() in extensions:
                yield DatasetFile(dataset_key=dataset_dir.name, path=file_path)


def read_table(path: Path) -> pd.DataFrame:
    """Read a supported tabular file into a dataframe."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def dataframe_to_documents(df: pd.DataFrame, dataset_key: str, file_path: Path) -> list[str]:
    """Turn a dataframe into compact Markdown chunks for retrieval."""
    docs: list[str] = []
    clean_df = df.copy()
    clean_df.columns = [str(c).strip() for c in clean_df.columns]

    summary = {
        "dataset": dataset_key,
        "file": file_path.name,
        "rows": int(len(clean_df)),
        "columns": list(clean_df.columns),
    }
    docs.append(
        "# Dataset summary\n"
        f"Source dataset: {dataset_key}\n"
        f"Source file: {file_path.name}\n"
        f"Rows: {len(clean_df)}\n"
        f"Columns: {', '.join(clean_df.columns)}\n"
        f"Machine-readable summary: {json.dumps(summary, ensure_ascii=False)}"
    )

    max_rows_per_doc = 25
    for start in range(0, len(clean_df), max_rows_per_doc):
        window = clean_df.iloc[start : start + max_rows_per_doc]
        records = window.fillna("").astype(str).to_dict(orient="records")
        body = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
        docs.append(
            "# World Cup 2026 tabular records\n"
            f"Source dataset: {dataset_key}\n"
            f"Source file: {file_path.name}\n"
            f"Rows represented: {start + 1}-{start + len(window)}\n"
            f"Records:\n{body}"
        )

    return docs


def write_seed_knowledge(base_dir: Path = KNOWLEDGE_BASE_DIR) -> None:
    """Write curated project facts that complement the Kaggle datasets."""
    base_dir.mkdir(parents=True, exist_ok=True)
    seed_path = base_dir / "fifa_world_cup_2026_seed_knowledge.md"
    seed_path.write_text(
        """# FIFA World Cup 2026 seed knowledge

The FIFA World Cup 2026 is hosted by Canada, Mexico, and the United States. FIFA describes
the tournament as a 48-team event with 104 matches across 16 host cities.

The tournament runs from Thursday, June 11, 2026 through Sunday, July 19, 2026. The opening
match is scheduled for Mexico City, and the final is scheduled for the New York New Jersey
venue.

Mexico is in Group A. Current public schedule references list Mexico's group as including
South Africa, Korea Republic, and Czechia. Treat schedule and draw facts as date-sensitive
and prefer the freshest official FIFA match schedule when updating the knowledge base.

Mexico group-stage schedule from FIFA public fixture references:
- Thursday, June 11, 2026: Mexico v South Africa, Group A, Mexico City Stadium. FIFA's
  schedule announcement listed kickoff at 13:00 local time in Mexico City.
- Thursday, June 18, 2026: Mexico v Korea Republic, Group A, Estadio Guadalajara.
- Wednesday, June 24, 2026: Czechia v Mexico, Group A, Mexico City Stadium.

Useful official FIFA references:
- FIFA World Cup 2026 fixtures and stadiums page
- FIFA article for Mexico fixtures and stadiums
- FIFA updated match schedule announcement

Projection guidance:
- Dataset-backed projections should explain which probability, ranking, or historical features
  were used.
- If a requested projection is not directly present in the data, present it as a simulation
  or scenario, not as a fact.
- Advancement from the group stage should model the 48-team format: 12 groups of 4, with the
  top two teams from each group plus the eight best third-place teams advancing to the round
  of 32.
""",
        encoding="utf-8",
    )


def build_knowledge_base(download: bool = True, force_download: bool = False) -> int:
    """Create Markdown knowledge files from Kaggle data plus seed facts."""
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    write_seed_knowledge(KNOWLEDGE_BASE_DIR)

    count = 1
    if download:
        try:
            download_kaggle_datasets(force=force_download)
        except Exception as exc:
            warning_path = KNOWLEDGE_BASE_DIR / "kaggle_download_warning.md"
            warning_path.write_text(
                "# Kaggle download warning\n"
                "The curated FIFA seed knowledge was still generated, but KaggleHub could "
                "not download the configured datasets.\n\n"
                f"Error: {exc}\n\n"
                "Configure Kaggle credentials, then rerun `wc2026-ingest` to add the full "
                "open-source datasets.\n",
                encoding="utf-8",
            )
            return count + 1

    for dataset_file in iter_dataset_files():
        try:
            df = read_table(dataset_file.path)
        except Exception as exc:
            error_path = KNOWLEDGE_BASE_DIR / f"{dataset_file.dataset_key}_{dataset_file.path.stem}_error.md"
            error_path.write_text(
                f"# Dataset read error\nSource: {dataset_file.path}\nError: {exc}\n",
                encoding="utf-8",
            )
            count += 1
            continue

        docs = dataframe_to_documents(df, dataset_file.dataset_key, dataset_file.path)
        for index, doc in enumerate(docs, start=1):
            out_path = KNOWLEDGE_BASE_DIR / f"{dataset_file.dataset_key}_{dataset_file.path.stem}_{index:04d}.md"
            out_path.write_text(doc, encoding="utf-8")
            count += 1

    return count
