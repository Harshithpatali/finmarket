# FinMarket AI — Global Stock Intelligence & Financial RAG

A research-oriented project combining global stock ML, financial RAG, news intelligence, FastAPI and Streamlit.

## ML
- Global pooled model across many tickers
- Logistic Regression
- XGBoost
- LightGBM
- LSTM
- Chronological validation and walk-forward evaluation
- SHAP explainability
- Simple research backtest

Initial target:
future 5-day return direction.

## RAG
Put legally obtained financial PDFs in `data/raw/`. The ingestion pipeline extracts text, chunks it, embeds it into FAISS and creates a deterministic starter knowledge graph.

## News
GNews is used for recent company news. Put `GNEWS_API_KEY` in `.env`.

## LLM
Groq is used only to synthesize supplied quantitative, news and retrieved-knowledge evidence. Put `GROQ_API_KEY` in `.env`.

## Local setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` from `.env.example` and add API keys.

Then:
```bash
python scripts/download_market_data.py
python scripts/build_features.py
python scripts/train_global_models.py
python scripts/walk_forward.py
python scripts/backtest_xgboost.py
python scripts/ingest_books.py
uvicorn api.main:app --reload
streamlit run app/streamlit_app.py
```

For Streamlit Cloud, set `API_URL` to the deployed FastAPI URL.

## Deployment
FastAPI can be deployed on Render with:
Build command:
```
pip install -r requirements.txt
```
Start command:
```
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Do not commit `.env`, PDFs, datasets or generated model artifacts.

This is a research/decision-support application, not personalized financial advice. Model probabilities are uncertain and should be interpreted with evidence and limitations.
