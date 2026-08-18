# src/odds_api.py
import requests
import os
from typing import Dict, Optional

class OddsAPI:
    def __init__(self):
        self.api_key = os.getenv("ODDS_API_KEY")
        self.base_url = "https://api.the-odds-api.com/v4"
        
    def get_match_odds(self, home_team: str, away_team: str, league: str = "soccer_italy_serie_a") -> Dict:
        """
        Fetch real odds for a specific match
        """
        if not self.api_key:
            return {"error": "No API key found"}
        
        try:
            # Get upcoming odds
            params = {
                "apiKey": self.api_key,
                "regions": "eu,uk",        # or "us" if you want American books
                "markets": "h2h,btts,over_under",
                "oddsFormat": "decimal"
            }
            
            response = requests.get(
                f"{self.base_url}/sports/{league}/odds", 
                params=params
            )
            
            if response.status_code != 200:
                return {"error": response.text}
                
            data = response.json()
            
            # Find the specific match
            for event in data:
                if (home_team.lower() in event['home_team'].lower() and 
                    away_team.lower() in event['away_team'].lower()):
                    return event  # Contains bookmakers + odds
                    
            return {"error": "Match not found in current odds"}
            
        except Exception as e:
            return {"error": str(e)}