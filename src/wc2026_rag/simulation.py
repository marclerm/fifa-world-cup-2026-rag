"""Small deterministic simulation helpers for World Cup 2026 scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MatchResult:
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int


def team_strength_from_rows(df: pd.DataFrame) -> dict[str, float]:
    """Infer a rough team-strength map from numeric fields in any imported dataset."""
    if df.empty:
        return {}

    team_cols = [c for c in df.columns if "team" in c.lower() or "country" in c.lower()]
    numeric_cols = list(df.select_dtypes(include="number").columns)
    if not team_cols or not numeric_cols:
        return {}

    team_col = team_cols[0]
    score_col = numeric_cols[0]
    grouped = df[[team_col, score_col]].dropna().groupby(team_col)[score_col].mean()
    values = grouped.astype(float)
    if values.max() == values.min():
        return {str(team): 1.0 for team in values.index}
    normalized = 0.5 + (values - values.min()) / (values.max() - values.min())
    return {str(team): float(score) for team, score in normalized.items()}


def simulate_group(
    teams: list[str],
    strengths: dict[str, float] | None = None,
    seed: int = 7,
) -> tuple[pd.DataFrame, list[MatchResult]]:
    """Simulate a four-team group table using simple strength-weighted Poisson scores."""
    rng = np.random.default_rng(seed)
    strengths = strengths or {}
    table = {
        team: {"team": team, "played": 0, "points": 0, "gf": 0, "ga": 0, "gd": 0}
        for team in teams
    }
    results: list[MatchResult] = []

    for home, away in combinations(teams, 2):
        home_strength = strengths.get(home, 1.0)
        away_strength = strengths.get(away, 1.0)
        home_goals = int(rng.poisson(1.25 * home_strength / max(away_strength, 0.2)))
        away_goals = int(rng.poisson(1.10 * away_strength / max(home_strength, 0.2)))
        results.append(MatchResult(home, away, home_goals, away_goals))

        table[home]["played"] += 1
        table[away]["played"] += 1
        table[home]["gf"] += home_goals
        table[home]["ga"] += away_goals
        table[away]["gf"] += away_goals
        table[away]["ga"] += home_goals

        if home_goals > away_goals:
            table[home]["points"] += 3
        elif away_goals > home_goals:
            table[away]["points"] += 3
        else:
            table[home]["points"] += 1
            table[away]["points"] += 1

    rows = []
    for team, row in table.items():
        row["gd"] = row["gf"] - row["ga"]
        rows.append(row)
    standings = pd.DataFrame(rows).sort_values(
        by=["points", "gd", "gf"], ascending=[False, False, False]
    )
    return standings.reset_index(drop=True), results


def describe_advancement(standings: pd.DataFrame) -> str:
    """Summarize group advancement under the 2026 format."""
    if standings.empty:
        return "No standings are available yet."
    top_two = standings.head(2)["team"].tolist()
    third = standings.iloc[2]["team"] if len(standings) >= 3 else None
    message = f"Projected automatic qualifiers: {', '.join(top_two)}."
    if third:
        message += f" {third} would need to compare well against other third-place teams."
    return message

