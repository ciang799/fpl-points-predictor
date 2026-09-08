"""
FPL Points Prediction Project — Step 2 (Simple Version)
------------------------------------------------------------------
Since only 1 gameweek has been played, there's no history to build
"form" features from yet. Instead, this script predicts a player's
points using their UNDERLYING performance stats (how threatening,
creative, and involved they were) plus cost and position.

This is a good foundation for your CV project now. Once more
gameweeks are played, swap to fpl_model.py which uses rolling form.

Run this locally (needs internet access):
    py -m pip install requests pandas scikit-learn matplotlib
    py fpl_simple_model.py
"""

import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

DATA_DIR = Path("fpl_data")


def load_players():
    players_path = DATA_DIR / "players.csv"
    if not players_path.exists():
        raise FileNotFoundError(
            "players.csv not found. Run fpl_data_explore.py first to create it."
        )
    return pd.read_csv(players_path)


def prepare_data(players_df):
    # Only look at players who've actually played minutes -- players with
    # 0 minutes have no meaningful stats and would just add noise
    df = players_df[players_df["minutes"] > 0].copy()

    # Numeric columns sometimes load as text from the API -- force them to numbers
    numeric_cols = ["now_cost", "total_points", "minutes", "influence",
                     "creativity", "threat", "ict_index", "bps", "bonus"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols)

    # One-hot encode position (GK/DEF/MID/FWD) so the model can use it
    df = pd.get_dummies(df, columns=["position"], prefix="pos")

    return df


def train_model(df):
    feature_cols = ["now_cost", "minutes", "influence", "creativity",
                     "threat", "bps"]
    feature_cols += [c for c in df.columns if c.startswith("pos_")]

    X = df[feature_cols]
    y = df["total_points"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    print(f"\nModel trained on {len(X_train)} players, tested on {len(X_test)}.")
    print(f"Mean Absolute Error: {mae:.2f} points")
    print("(On average, predictions are off by this many points.)")
    print("Note: with only 1 gameweek played, this is a rough first pass --")
    print("accuracy will improve a lot once you retrain with more gameweeks.")

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("\nWhich features matter most:")
    print(importances.sort_values(ascending=False).to_string())

    return model, feature_cols


def show_value_picks(df, model, feature_cols):
    """Players whose actual points beat what the model expected from their
    cost -- i.e. possible 'good value' picks based on underlying stats."""
    df = df.copy()
    df["predicted_points"] = model.predict(df[feature_cols])
    df["overperformance"] = df["total_points"] - df["predicted_points"]

    print("\nTop 10 players outperforming their underlying stats (possible bargains):")
    cols_to_show = ["web_name", "team", "total_points", "now_cost",
                     "predicted_points", "overperformance"]
    print(df.sort_values("overperformance", ascending=False)[cols_to_show]
          .head(10).round(1).to_string(index=False))


def main():
    print("Loading players.csv...")
    players_df = load_players()

    print("Preparing data (filtering to players with minutes played)...")
    df = prepare_data(players_df)
    print(f"{len(df)} players have played minutes so far.")

    if len(df) < 20:
        print("\nNot enough players with data yet -- try again after gameweek 2.")
        return

    print("\nTraining model...")
    model, feature_cols = train_model(df)

    show_value_picks(df, model, feature_cols)


if __name__ == "__main__":
    main()
