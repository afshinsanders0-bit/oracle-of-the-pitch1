# test_analyst.py
from src.analyst import FootballAnalyst
from dotenv import load_dotenv
import os

load_dotenv()

analyst = FootballAnalyst(provider="groq")

test_match = {
    "home_team": "Napoli",
    "away_team": "Bologna",
    "league": "Serie A",
    "date": "2026-05-11"
}

model_preds = {
    "home_win_prob": 0.58,
    "draw_prob": 0.25,
    "away_win_prob": 0.17,
    "btts_prob": 0.62,
    "over_2_5_prob": 0.55,
    "expected_goals": 2.8
}

print("🔄 Generating analyst preview...")
preview = analyst.generate_preview(test_match, model_preds)
print("\n" + "="*60)
print(preview)
print("="*60)