import re

POSITIVE = {
    "growth","profit","profits","strong","surge","rise","rises","positive","beat",
    "beats","upgrade","bullish","record","gain","gains","expansion","outperform","improve","improved"
}
NEGATIVE = {
    "loss","losses","fall","falls","decline","negative","miss","misses","downgrade",
    "bearish","risk","risks","fraud","weak","weakness","drop","drops","lawsuit","warning"
}

def simple_sentiment(text: str) -> float:
    words = re.findall(r"[a-z]+", text.lower())
    if not words: return 0.0
    pos = sum(w in POSITIVE for w in words)
    neg = sum(w in NEGATIVE for w in words)
    return (pos - neg) / max(1, len(words))

def aggregate_news(articles):
    if not articles:
        return {"news_count":0,"news_sentiment":0.0,"positive_news":0,"negative_news":0}
    scores = []
    for a in articles:
        text = " ".join([a.get("title",""), a.get("description",""), a.get("content","")])
        scores.append(simple_sentiment(text))
    return {
        "news_count": len(articles),
        "news_sentiment": sum(scores)/len(scores),
        "positive_news": sum(s > 0 for s in scores),
        "negative_news": sum(s < 0 for s in scores),
    }
