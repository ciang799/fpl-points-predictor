"""
FPL Points Prediction Project — Step 2: Build a Prediction Model
------------------------------------------------------------------
This script:
  1. Loads the players.csv you already created (run fpl_data_explore.py first)
  2. Fetches each player's gameweek-by-gameweek history from the FPL API
  3. Builds "rolling form" features (average points/minutes over recent games)
  4. Trains a model to predict a player's points in their NEXT gameweek
  5. Reports how accurate the model is, and which features matter most

Run this locally (needs internet access):
    py -m pip install requests pandas scikit-learn matplotlib
    py fpl_model.py

This will take a minute or two — it has to fetch history for ~600 players.
"""

import requests
import pandas as pd
import time
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

BASE_URL = "https://fantasy.premierleague.com/api"
DATA_DIR = Path("fpl_data")
DATA_DIR.mkdir(exist_ok=True)

ROLLING_WINDOW = 3  # how many past games to average over for "form" features


def load_players():
    """Load the players.csv created by fpl_data_explore.py"""
    players_path = DATA_DIR / "players.csv"
    if not players_path.exists():
        raise FileNotFoundError(
            "players.csv not found. Run fpl_data_explore.py first to create it."
        )
    return pd.read_csv(players_path)


def fetch_all_histories(player_ids):
    """
    Fetch gameweek-by-gameweek history for every player.
    This is one API call per player, so it takes a little while.
    """
    all_rows = []
    total = len(player_ids)
    for i, pid in enumerate(player_ids, 1):
        try:
            r = requests.get(f"{BASE_URL}/element-summary/{pid}/")
            r.raise_for_status()
            history = r.json().get("history", [])
            for gw in history:
                gw["player_id"] = pid
                all_rows.append(gw)
        except Exception as e:
            print(f"  Skipped player {pid}: {e}")

        if i % 50 == 0 or i == total:
            print(f"  Fetched {i}/{total} players...")
        time.sleep(0.02)  # be polite to the API

    return pd.DataFrame(all_rows)


def build_features(history_df, players_df):
    """
    Turn raw gameweek history into a training dataset:
    features = form BEFORE a gameweek, target = points IN that gameweek.
    """
    history_df = history_df.sort_values(["player_id", "round"])

    # Rolling average of points and minutes over the previous N games
    # (shift(1) so we only use PAST games, never the game we're predicting)
    grouped = history_df.groupby("player_id")
    history_df["form_points"] = (
        grouped["total_points"]
        .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
    )
    history_df["form_minutes"] = (
        grouped["minutes"]
        .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
    )
    history_df["form_bonus"] = (
        grouped["bonus"]
        .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
    )

    # Merge in static player info: position, cost, team
    meta = players_df[["id", "position", "now_cost"]].rename(columns={"id": "player_id"})
    history_df = history_df.merge(meta, on="player_id", how="left")

    # One-hot encode position (DEF/MID/FWD/GK) so the model can use it
    history_df = pd.get_dummies(history_df, columns=["position"], prefix="pos")

    # Drop rows with no prior form (first game for each player has nothing to average)
    history_df = history_df.dropna(subset=["form_points", "form_minutes", "form_bonus"])

    return history_df


def train_model(history_df):
    feature_cols = ["form_points", "form_minutes", "form_bonus", "was_home", "now_cost"]
    feature_cols += [c for c in history_df.columns if c.startswith("pos_")]

    X = history_df[feature_cols]
    y = history_df["total_points"]  # this gameweek's actual points = what we predict

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    print(f"\nModel trained on {len(X_train)} gameweek records, tested on {len(X_test)}.")
    print(f"Mean Absolute Error: {mae:.2f} points")
    print("(On average, predictions are off by this many points per gameweek.)")

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("\nWhich features matter most:")
    print(importances.sort_values(ascending=False).to_string())

    return model, feature_cols


def main():
    print("Loading players.csv...")
    players_df = load_players()

    print(f"Fetching gameweek history for {len(players_df)} players (this takes a minute)...")
    history_df = fetch_all_histories(players_df["id"].tolist())
    history_df.to_csv(DATA_DIR / "gameweek_history.csv", index=False)
    print(f"Saved raw history to {DATA_DIR / 'gameweek_history.csv'}")

    print("\nBuilding features (rolling form, position, cost)...")
    feature_df = build_features(history_df, players_df)
    feature_df.to_csv(DATA_DIR / "training_data.csv", index=False)
    print(f"Saved training dataset ({len(feature_df)} rows) to {DATA_DIR / 'training_data.csv'}")

    if len(feature_df) < 50:
        print("\nNot enough gameweek data yet this season to train a reliable model.")
        print("Try again in a few weeks once more gameweeks have been played.")
        return

    print("\nTraining model...")
    train_model(feature_df)


if __name__ == "__main__":
    main()
