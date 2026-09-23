import os
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
load_dotenv()

GNEWS_URL = "https://gnews.io/api/v4/search"

def search_gnews(query: str, max_results: int = 10, days: int = 3):
    api_key = os.getenv("GNEWS_API_KEY")
    if not api_key:
        raise RuntimeError("GNEWS_API_KEY is missing from .env")
    from_dt = datetime.now(timezone.utc) - timedelta(days=days)
    params = {
        "q": query, "lang": "en", "max": min(max_results, 10),
        "from": from_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sortby": "publishedAt", "apikey": api_key,
    }
    r = requests.get(GNEWS_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("articles", [])
