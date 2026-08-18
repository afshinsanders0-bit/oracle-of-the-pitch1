#!/usr/bin/env python3
"""
Test your football-data.org API key
Quickly validates if the key works and what data you can access
"""

import os
import sys
from pathlib import Path
import requests

# Load API key from .env
def load_api_key():
    env_locations = [
        Path(".env"),
        Path.home() / ".env",
    ]
    for env_file in env_locations:
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("FOOTBALL_DATA_KEY="):
                    val = line.split("=", 1)[1].strip().strip("\"'")
                    if val and val != "your_key_here":
                        print(f"✓ Found API key in {env_file}")
                        return val
    
    # Check environment variable
    key = os.environ.get("FOOTBALL_DATA_KEY", "").strip()
    if key:
        print("✓ Found API key in environment variable")
        return key
    
    return None


def test_api_key(api_key):
    """Test if the API key works"""
    if not api_key:
        print("✗ No API key found!")
        print("  Add to .env: FOOTBALL_DATA_KEY=your_key_here")
        return False
    
    print(f"\nTesting API key: {api_key[:8]}...{api_key[-4:]}")
    print("-" * 60)
    
    headers = {"X-Auth-Token": api_key}
    
    # Test 1: Basic validation
    print("\n1️⃣  Testing basic API access...")
    try:
        r = requests.get(
            "https://api.football-data.org/v4/competitions/PL/matches?season=2024&limit=1&status=FINISHED",
            headers=headers,
            timeout=10
        )
        
        if r.status_code == 200:
            print("   ✓ API key is VALID")
            data = r.json()
            matches = data.get("matches", [])
            if matches:
                m = matches[0]
                print(f"   ✓ Can fetch Premier League data")
                print(f"     Sample: {m['homeTeam']['name']} vs {m['awayTeam']['name']}")
            return True
        elif r.status_code == 403:
            print("   ✗ API key INVALID or EXPIRED")
            return False
        elif r.status_code == 429:
            print("   ✗ Rate limited (too many requests)")
            return False
        else:
            print(f"   ✗ HTTP {r.status_code}")
            print(f"   Response: {r.text[:200]}")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def test_plan_coverage(api_key):
    """Check which leagues your plan covers"""
    print("\n2️⃣  Checking plan coverage...")
    
    leagues = {
        "EPL":       {"code": "PL",  "name": "Premier League"},
        "LA_LIGA":   {"code": "PD",  "name": "La Liga"},
        "SERIE_A":   {"code": "SA",  "name": "Serie A"},
        "BUNDESLIGA":{"code": "BL1", "name": "Bundesliga"},
        "LIGUE_1":   {"code": "FL1", "name": "Ligue 1"},
        "ENG_CHAMP": {"code": "ELC", "name": "Championship"},
    }
    
    headers = {"X-Auth-Token": api_key}
    
    for key, league in leagues.items():
        try:
            r = requests.get(
                f"https://api.football-data.org/v4/competitions/{league['code']}/matches?season=2024&limit=1",
                headers=headers,
                timeout=5
            )
            if r.status_code == 200:
                print(f"   ✓ {league['name']:<20} AVAILABLE")
            elif r.status_code == 403:
                print(f"   ✗ {league['name']:<20} NOT IN YOUR PLAN")
        except:
            pass


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Football-Data.org API Key Validator")
    print("="*60)
    
    api_key = load_api_key()
    
    if test_api_key(api_key):
        test_plan_coverage(api_key)
        print("\n" + "="*60)
        print("✓ Your API key is working!")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("✗ API key test FAILED")
        print("="*60)
        print("\nSolutions:")
        print("1. Get a FREE key: https://www.football-data.org/client/register")
        print("2. Add to .env: echo 'FOOTBALL_DATA_KEY=your_key' > .env")
        print("3. Run this script again to verify\n")