# Credit Scoring Demo (Streamlit)

A minimal, ready-to-run project that extracts company metadata from the web and computes a basic credit score, with an exportable PDF report.

## Features
- Google Custom Search → official website
- Scrape website → extract full text
- OpenAI LLM → normalize metadata (2-sentence description)
- Pillars: Financials, Governance, Industry, Country, Reputation (news sentiment)
- Download **PDF** report

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Configure Keys
Edit `backend.py` and replace these placeholders at the top:
```python
OPENAI_API_KEY = "OPENAI_KEY_REPLACE_ME"
GOOGLE_API_KEY = "GOOGLE_KEY_REPLACE_ME"
GOOGLE_CX      = "GOOGLE_CX_REPLACE_ME"
```
Or export as environment variables instead.

## Notes
- For companies with little/no online presence, Reputation (news) defaults to 5.0 (neutral).
- This is a demo heuristic; refine pillars/weights for production.
