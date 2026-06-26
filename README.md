# 🔎 DataScout — Semantic Dataset Search Engine

> Search datasets by **meaning**, not by title.

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces/alinawazmahar/DataScout)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-5.x-orange)](https://gradio.app/)
[![Model](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-purple)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

---

## The Problem

Most dataset search tools match your *words*. Type `"audio classification"` and you get every dataset containing those two words — useful or not.

DataScout matches your *intent*. Describe what you need in plain English and it finds datasets that are semantically close to what you actually mean.

---

## How It Works

```
Your query (plain English)
        │
        ▼
HuggingFace Hub API ──► ~40 keyword-filtered candidates
        │
        ▼
Embed query + all candidate descriptions   (all-MiniLM-L6-v2)
        │
        ▼
Cosine similarity  ──► semantic closeness score
        │
        ▼
Quality score      ──► log-scaled downloads + likes + recency + metadata
        │
        ▼
Final score = 0.75 × similarity + 0.25 × quality
        │
        ▼
Top-K results ranked and displayed
```

### Quality Score Breakdown

The quality signal is a **documentation/popularity proxy** — not a verified data-quality audit. It blends:

| Signal | Weight | Notes |
|--------|--------|-------|
| Downloads | 40% | Log-scaled — old popular datasets don't dominate |
| Likes | 20% | Log-scaled with cap at 1,000 |
| Recency | 15% | Full credit if updated within 90 days, decays over 3 years |
| Metadata completeness | 25% | Has real description (70%) + has tags (30%) |

Always inspect a dataset's card before using it in production.

---

## Demo

🚀 **[Try it live on HuggingFace Spaces](https://huggingface.co/spaces/alinawazmahar/DataScout)**

Example queries to try:
- `"sensor data from industrial machines with timestamps"`
- `"Sindhi language text for sentiment analysis"`
- `"medical images for cancer detection"`
- `"satellite imagery for crop monitoring"`
- `"conversational dialogue data for chatbots"`

---

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/alinawazmahar/DataScout.git
cd DataScout

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

The app will be available at `http://localhost:7860`

---

## Enabling Kaggle Search

Kaggle search is implemented but disabled by default. To activate:

1. Go to `kaggle.com` → Profile → **Settings** → **API** → **Create New Token**
   This downloads `kaggle.json`

2. Set environment variables:
   ```bash
   export KAGGLE_USERNAME=your_username
   export KAGGLE_KEY=your_api_key
   ```
   Or add them as **Secrets** in your HuggingFace Space settings

3. The "Search Kaggle" checkbox will activate automatically on next launch

> ⚠️ Never commit `kaggle.json` to Git — it's in `.gitignore` for this reason

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Dataset discovery | `huggingface_hub` API |
| Similarity ranking | `numpy` cosine similarity |
| UI | `gradio` |
| Optional source | `kaggle` API |

---

## Project Structure

```
DataScout/
├── app.py              # Main application — search pipeline + Gradio UI
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

---

## Limitations (v1)

- Only HuggingFace + Kaggle are wired up (NASA, UCI, GitHub Datasets would need scraping — a natural v2 extension)
- Live search only — no pre-built index, so each query re-fetches from the Hub
- "Quality" is a metadata proxy, not a verified data-quality audit

---

## Roadmap

- [ ] Pre-built embedding index for faster search
- [ ] NASA + UCI repository support
- [ ] Actual data profiling (column types, missing values, class balance)
- [ ] Filter by task type, language, license
- [ ] Dataset comparison side-by-side

---

## Author

**Ali Nawaz Mahar**
BS Computer Science — Shaikh Ayaz University, Shikarpur

[![HuggingFace](https://img.shields.io/badge/🤗-alinawazmahar-yellow)](https://huggingface.co/alinawazmahar)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/alinawazmahar)

---

## License

MIT License — feel free to use, modify, and build on this.
