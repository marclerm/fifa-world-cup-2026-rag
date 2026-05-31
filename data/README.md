# Local Data Layout

`data/raw/` is for local source files used by ingestion. It is gitignored so full dataset
downloads do not accidentally get committed.

Current local folders:

- `match_probability_baseline/`
  - `future_match_probabilities.csv`: baseline win/draw/loss probabilities and ELO-style
    features for future World Cup 2026 group matches.
- `match_schedule_unofficial/`
  - `matches.csv`: match numbers, team IDs, city IDs, stage IDs, kickoff timestamps, and labels.
  - `teams.csv`: team names, FIFA codes, group letters, and placeholder flags.
  - `host_cities.csv`: host city, country, venue, region, and airport metadata.
  - `tournament_stages.csv`: stage names and order.
  - `worldcup2026.sqlite`: local SQLite copy from the same schedule dataset.
- `team_historical_outcomes/`
  - `world_cup_team_history_train.csv`: historical team features and outcomes.
  - `world_cup_team_features_2026_test.csv`: 2026 team features for projection work.

Run `wc2026-ingest` after changing these files.

