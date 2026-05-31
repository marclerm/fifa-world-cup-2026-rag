from wc2026_rag.simulation import describe_advancement, simulate_group


def test_simulate_group_returns_standings_and_results():
    teams = ["Mexico", "South Africa", "South Korea", "Czechia"]
    standings, results = simulate_group(teams, seed=1)

    assert len(standings) == 4
    assert len(results) == 6
    assert set(standings["team"]) == set(teams)
    assert "Projected automatic qualifiers" in describe_advancement(standings)

