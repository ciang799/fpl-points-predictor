# FPL Points Predictor

Predicts Fantasy Premier League player points using rolling form, cost, and position, built with Python, pandas, and scikit-learn.

## What it does

This project pulls live player and gameweek data from the free [FPL API](https://fantasy.premierleague.com/api/bootstrap-static/), engineers features around recent player form, and trains a machine learning model to predict how many points a player will score in their next gameweek.

## Project structure

- **`fpl_data_explore.py`** — fetches player, team, and fixture data from the FPL API and builds a clean `players.csv` for exploration.
- **`fpl_model.py`** — fetches gameweek-by-gameweek history for every player, builds rolling "form" features (average points/minutes/bonus over the last 3 gameweeks), and trains a Random Forest model to predict next-gameweek points.
- **`fpl_simple_model.py`** — an earlier, simpler version built when only 1 gameweek had been played (no history available yet). Kept in the repo as a record of an important lesson (see below).

## How to run it

```bash
pip install requests pandas scikit-learn matplotlib
python fpl_data_explore.py   # builds players.csv
python fpl_model.py          # builds the prediction model
```

## Results

After 3 gameweeks of the season, the model (`fpl_model.py`) achieved:

- **Mean Absolute Error: ~1.03 points** per gameweek on held-out test data
- Most important features: recent minutes played (form_minutes), player cost, and recent points form

## A lesson in data leakage

The first version of this project (`fpl_simple_model.py`) reported a suspiciously low error (~0.72 points) that turned out to be misleading. On inspection, one input feature (`bps`, the Bonus Points System score) is itself used by the FPL scoring system to calculate part of a player's total points — meaning the model was partly "cheating" by using information that wouldn't be available before a gameweek happens.

The fix was to rebuild the feature set (`fpl_model.py`) so that every feature is calculated only from **past** gameweeks (using a time-shifted rolling average), never from the gameweek being predicted. This is a more honest test of whether the model can actually forecast the future rather than just reconstruct the present.

## Possible next steps

- Incorporate fixture difficulty (upcoming opponent strength) as a feature
- Track prediction accuracy across a full season as more data accumulates
- Build a simple "best XI" or transfer-recommendation tool on top of the model
- Add visualizations (predicted vs. actual points, feature importance chart)

## Tech used

Python, pandas, scikit-learn (Random Forest Regressor), the FPL public API
