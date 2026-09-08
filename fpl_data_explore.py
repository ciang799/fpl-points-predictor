"""
FPL Points Prediction Project — Step 1: Data Collection & Exploration
------------------------------------------------------------------
Pulls data from the free, public Fantasy Premier League API and does
a first pass of exploration. Run this locally (not in a restricted
sandbox) since it needs internet access to fantasy.premierleague.com.

Usage:
    pip install requests pandas matplotlib
    python fpl_data_explore.py
"""

import requests
import pandas as pd
import json
from pathlib import Path

BASE_URL = "https://fantasy.premierleague.com/api"
DATA_DIR = Path("fpl_data")
DATA_DIR.mkdir(exist_ok=True)


def fetch_bootstrap():
    """Main endpoint: players, teams, positions, gameweeks."""
    r = requests.get(f"{BASE_URL}/bootstrap-static/")
    r.raise_for_status()
    data = r.json()
    with open(DATA_DIR / "bootstrap.json", "w") as f:
        json.dump(data, f)
    return data


def fetch_player_history(player_id):
    """Per-gameweek history for a single player."""
    r = requests.get(f"{BASE_URL}/element-summary/{player_id}/")
    r.raise_for_status()
    return r.json()


def fetch_fixtures():
    """All fixtures (for difficulty ratings, home/away, etc.)."""
    r = requests.get(f"{BASE_URL}/fixtures/")
    r.raise_for_status()
    data = r.json()
    with open(DATA_DIR / "fixtures.json", "w") as f:
        json.dump(data, f)
    return data


def build_players_df(bootstrap):
    players = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])[["id", "name", "short_name"]]
    positions = pd.DataFrame(bootstrap["element_types"])[["id", "singular_name", "singular_name_short"]]

    players = players.merge(teams, left_on="team", right_on="id", suffixes=("", "_team"))
    players = players.merge(positions, left_on="element_type", right_on="id", suffixes=("", "_pos"))

    # Keep the columns most useful for a points-prediction project
    cols = [
        "id", "web_name", "first_name", "second_name",
        "name", "singular_name_short",  # team, position
        "now_cost", "total_points", "points_per_game",
        "form", "minutes", "goals_scored", "assists",
        "clean_sheets", "goals_conceded", "own_goals",
        "penalties_saved", "penalties_missed", "yellow_cards",
        "red_cards", "saves", "bonus", "bps", "influence",
        "creativity", "threat", "ict_index", "selected_by_percent",
        "transfers_in_event", "transfers_out_event", "status",
    ]
    players_df = players[cols].rename(columns={
        "name": "team", "singular_name_short": "position"
    })
    return players_df


def main():
    print("Fetching bootstrap data (players, teams, positions)...")
    bootstrap = fetch_bootstrap()

    print("Fetching fixtures...")
    fixtures = fetch_fixtures()

    print("Building players dataframe...")
    players_df = build_players_df(bootstrap)
    players_df.to_csv(DATA_DIR / "players.csv", index=False)

    print(f"\nSaved {len(players_df)} players to {DATA_DIR / 'players.csv'}")
    print("\n--- Quick exploration ---")
    print("\nTop 10 by total points:")
    print(players_df.sort_values("total_points", ascending=False)
          [["web_name", "team", "position", "total_points", "now_cost"]]
          .head(10).to_string(index=False))

    print("\nPoints per game leaders (min 500 minutes):")
    active = players_df[players_df["minutes"].astype(float) > 500].copy()
    active["points_per_game"] = active["points_per_game"].astype(float)
    print(active.sort_values("points_per_game", ascending=False)
          [["web_name", "team", "position", "points_per_game"]]
          .head(10).to_string(index=False))

    print("\nColumns available for feature engineering:")
    print(list(players_df.columns))

    print(f"\nDone. Data cached in ./{DATA_DIR}/ for reuse.")


if __name__ == "__main__":
    main()
