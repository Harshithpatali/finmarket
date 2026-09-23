import json, os
from dotenv import load_dotenv
from groq import Groq
load_dotenv()

SYSTEM_PROMPT = """
You are a financial research assistant. Produce evidence-grounded analysis, not certainty.
Separate observed market facts, quantitative model outputs, retrieved financial knowledge,
current news, interpretation and uncertainty. Do not invent prices, news, sources, model
results, or citations. A model probability is not a guarantee of future performance.
"""

def analyze(context: dict, model: str = "llama-3.3-70b-versatile"):
    key = os.getenv("GROQ_API_KEY")
    if not key: raise RuntimeError("GROQ_API_KEY is missing from .env")
    client = Groq(api_key=key)
    response = client.chat.completions.create(
        model=model, temperature=0.1,
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":json.dumps(context, default=str, indent=2)},
        ],
    )
    return response.choices[0].message.content
