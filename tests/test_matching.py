"""
Quick test script for the /match endpoint -- avoids all the PowerShell/cmd
quote-escaping pain of building the same request with curl.

Usage:
    python test_match.py
"""
import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SERVICE_API_KEY")
URL = "http://localhost:8000/match"

if not API_KEY:
    raise SystemExit("SERVICE_API_KEY not found in .env")

payload = {
    "title": "Test Film",
    "overview": "A sci-fi thriller about time travel",
    "genres": ["Sci-Fi", "Thriller"],
    "avg_rating": 4.0,
    "rating_count": 50,
    "popularity": 20,
}

response = requests.post(
    URL,
    json=payload,
    headers={"x-api-key": API_KEY},
)

print(f"Status: {response.status_code}")
print(response.json())