"""
Smart Dataset Search Engine
----------------------------
Search datasets by MEANING, not by title.

Pipeline:
  1. User types a natural-language query.
  2. We pull a candidate pool of datasets from HuggingFace Hub
     (keyword pre-filter, so we don't have to embed the whole Hub).
  3. We embed the query + each candidate's description/card text.
  4. We rank candidates by a blend of:
       - semantic similarity (cosine sim of embeddings)
       - a "quality" signal (downloads, likes, recency, metadata completeness)
  5. We show the top results with a quality badge and a link.

Kaggle support is stubbed in (see search_kaggle) so it can be enabled later
by adding KAGGLE_USERNAME / KAGGLE_KEY as Space secrets.
"""

import os
import math
import datetime
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import gradio as gr
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, good enough for free CPU Spaces
HF_CANDIDATE_POOL = 40                  # how many HF datasets we pull per query before re-ranking
TOP_K_DEFAULT = 10

# Weight given to semantic similarity vs. quality score when ranking.
# similarity is the main driver; quality nudges ties and surfaces well-documented datasets.
SIMILARITY_WEIGHT = 0.75
QUALITY_WEIGHT = 0.25


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DatasetResult:
    source: str                # "HuggingFace" | "Kaggle"
    dataset_id: str
    title: str
    description: str
    url: str
    downloads: Optional[int] = None
    likes: Optional[int] = None
    last_modified: Optional[datetime.datetime] = None
    tags: list = field(default_factory=list)

    similarity: float = 0.0
    quality: float = 0.0
    final_score: float = 0.0


# ---------------------------------------------------------------------------
# Embedding model (loaded once, lazily, at first use)
# ---------------------------------------------------------------------------

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


def cosine_sim_matrix(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    # Vectors are already normalized, so dot product == cosine similarity.
    return doc_vecs @ query_vec


# ---------------------------------------------------------------------------
# HuggingFace search
# ---------------------------------------------------------------------------

_hf_api = HfApi()


def _extract_hf_description(card_data, tags: Optional[list]) -> str:
    """
    Pull the best available free-text description out of a HF DatasetInfo.
    card_data is a DatasetCardData object (or None) - it doesn't always carry
    free text, so we fall back to tags as a weak signal if nothing else exists.
    """
    text_parts = []

    if card_data is not None:
        desc = getattr(card_data, "description", None)
        if desc:
            text_parts.append(str(desc))
        pretty_name = getattr(card_data, "pretty_name", None)
        if pretty_name:
            text_parts.append(str(pretty_name))

    if not text_parts and tags:
        text_parts.append("Tags: " + ", ".join(tags[:15]))

    return " | ".join(text_parts) if text_parts else ""


def search_huggingface(query: str, pool_size: int = HF_CANDIDATE_POOL) -> list[DatasetResult]:
    """
    Pull a candidate pool of datasets from the HF Hub using its own keyword
    search, then return them as DatasetResult objects for re-ranking.
    """
    try:
        raw_results = list(
            _hf_api.list_datasets(
                search=query,
                limit=pool_size,
                full=True,
            )
        )
    except Exception as e:
        print(f"[HF search error] {e}")
        return []

    results = []
    for d in raw_results:
        if getattr(d, "private", False) or getattr(d, "disabled", False):
            continue

        description = _extract_hf_description(getattr(d, "card_data", None), d.tags)
        title = d.id.split("/")[-1].replace("-", " ").replace("_", " ")

        results.append(
            DatasetResult(
                source="HuggingFace",
                dataset_id=d.id,
                title=title,
                description=description or "(no description available)",
                url=f"https://huggingface.co/datasets/{d.id}",
                downloads=getattr(d, "downloads", None) or 0,
                likes=getattr(d, "likes", None) or 0,
                last_modified=getattr(d, "last_modified", None),
                tags=d.tags or [],
            )
        )
    return results


# ---------------------------------------------------------------------------
# Kaggle search (stub - activate once KAGGLE_USERNAME / KAGGLE_KEY are set)
# ---------------------------------------------------------------------------

def kaggle_configured() -> bool:
    return bool(os.environ.get("KAGGLE_USERNAME")) and bool(os.environ.get("KAGGLE_KEY"))


def search_kaggle(query: str, pool_size: int = 20) -> list[DatasetResult]:
    """
    Placeholder for Kaggle dataset search.

    To enable:
      1. Get kaggle.json from kaggle.com -> Settings -> API -> Create New Token
      2. In the HF Space settings, add secrets:
           KAGGLE_USERNAME = <username from kaggle.json>
           KAGGLE_KEY      = <key from kaggle.json>
      3. Add "kaggle" to requirements.txt
      4. Uncomment the implementation below.
    """
    if not kaggle_configured():
        return []

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        datasets = api.dataset_list(search=query)[:pool_size]

        results = []
        for ds in datasets:
            results.append(
                DatasetResult(
                    source="Kaggle",
                    dataset_id=ds.ref,
                    title=getattr(ds, "title", ds.ref),
                    description=getattr(ds, "subtitle", "") or "(no description available)",
                    url=f"https://www.kaggle.com/datasets/{ds.ref}",
                    downloads=getattr(ds, "downloadCount", 0),
                    likes=getattr(ds, "voteCount", 0),
                    last_modified=None,
                    tags=[t.name for t in getattr(ds, "tags", [])] if getattr(ds, "tags", None) else [],
                )
            )
        return results
    except Exception as e:
        print(f"[Kaggle search error] {e}")
        return []


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

def _log_scale(value: float, cap: float) -> float:
    """Squash an unbounded count (downloads/likes) into 0..1 using a log curve."""
    if value <= 0:
        return 0.0
    return min(1.0, math.log1p(value) / math.log1p(cap))


def compute_quality_score(d: DatasetResult) -> float:
    """
    A transparent, explainable proxy for dataset quality - NOT a measure of
    actual data cleanliness. Blends:
      - popularity (downloads, likes) - log-scaled
      - recency - newer datasets score slightly higher, with a long decay
      - metadata completeness - has a real description? has tags?
    """
    downloads_score = _log_scale(d.downloads or 0, cap=1_000_000)
    likes_score = _log_scale(d.likes or 0, cap=1_000)

    recency_score = 0.5
    if d.last_modified is not None:
        try:
            now = datetime.datetime.now(datetime.timezone.utc)
            last_mod = d.last_modified
            if last_mod.tzinfo is None:
                last_mod = last_mod.replace(tzinfo=datetime.timezone.utc)
            age_days = (now - last_mod).days
            recency_score = max(0.0, 1.0 - (age_days / (365 * 3)))
        except Exception:
            recency_score = 0.5

    has_real_description = 1.0 if (d.description and "no description" not in d.description) else 0.0
    has_tags = 1.0 if d.tags else 0.0
    completeness_score = (has_real_description * 0.7) + (has_tags * 0.3)

    quality = (
        0.40 * downloads_score
        + 0.20 * likes_score
        + 0.15 * recency_score
        + 0.25 * completeness_score
    )
    return round(quality, 4)


# ---------------------------------------------------------------------------
# Core search + ranking pipeline
# ---------------------------------------------------------------------------

def run_search(query: str, top_k: int, use_hf: bool, use_kaggle: bool) -> list[DatasetResult]:
    query = (query or "").strip()
    if not query:
        return []

    candidates: list[DatasetResult] = []
    if use_hf:
        candidates.extend(search_huggingface(query))
    if use_kaggle:
        candidates.extend(search_kaggle(query))

    if not candidates:
        return []

    texts = [query] + [f"{c.title}. {c.description}" for c in candidates]
    embeddings = embed_texts(texts)
    query_vec, doc_vecs = embeddings[0], embeddings[1:]

    similarities = cosine_sim_matrix(query_vec, doc_vecs)

    for c, sim in zip(candidates, similarities):
        c.similarity = float(sim)
        c.quality = compute_quality_score(c)
        sim_clamped = max(0.0, c.similarity)
        c.final_score = (SIMILARITY_WEIGHT * sim_clamped) + (QUALITY_WEIGHT * c.quality)

    candidates.sort(key=lambda c: c.final_score, reverse=True)
    return candidates[:top_k]


# ---------------------------------------------------------------------------
# UI rendering
# ---------------------------------------------------------------------------

def quality_badge(score: float) -> str:
    if score >= 0.66:
        return "🟢 High"
    if score >= 0.33:
        return "🟡 Medium"
    return "🔴 Low"


def format_results_markdown(results: list[DatasetResult]) -> str:
    if not results:
        return (
            "### No results yet\n\n"
            "Try a query like *\"sensor data from industrial machines with timestamps\"* "
            "or *\"Sindhi text for sentiment analysis\"*."
        )

    lines = [f"### Found {len(results)} result(s)\n"]
    for i, r in enumerate(results, 1):
        downloads = f"{r.downloads:,}" if r.downloads is not None else "—"
        likes = f"{r.likes:,}" if r.likes is not None else "—"
        desc = r.description
        if len(desc) > 280:
            desc = desc[:280].rsplit(" ", 1)[0] + "…"

        lines.append(
            f"**{i}. [{r.title}]({r.url})**  \n"
            f"`{r.source}` · Quality: {quality_badge(r.quality)} "
            f"(match {r.similarity:.2f} · quality {r.quality:.2f}) · "
            f"⬇ {downloads} · ❤ {likes}  \n"
            f"{desc}\n"
        )
    return "\n".join(lines)


def search_handler(query: str, top_k: int, use_hf: bool, use_kaggle: bool):
    if not use_hf and not use_kaggle:
        return "### Pick at least one source (HuggingFace or Kaggle) above."

    results = run_search(query, int(top_k), use_hf, use_kaggle)
    return format_results_markdown(results)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

EXAMPLE_QUERIES = [
    "sensor data from industrial machines with timestamps",
    "Sindhi language text for sentiment analysis",
    "medical images for cancer detection",
    "satellite imagery for crop monitoring",
    "conversational dialogue data for chatbots",
]

with gr.Blocks(title="Smart Dataset Search Engine", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔎 Smart Dataset Search Engine
        Search datasets by **meaning**, not by title.

        Type what you need in plain English — the engine semantically matches your
        query against dataset descriptions (not just keywords in the name) and
        ranks results by relevance + a quality signal (popularity, recency, metadata
        completeness).
        """
    )

    with gr.Row():
        query_box = gr.Textbox(
            label="Describe the dataset you need",
            placeholder='e.g. "sensor data from industrial machines with timestamps"',
            scale=4,
        )
        top_k_slider = gr.Slider(
            minimum=3, maximum=20, value=TOP_K_DEFAULT, step=1, label="Results", scale=1
        )

    with gr.Row():
        use_hf_checkbox = gr.Checkbox(value=True, label="Search HuggingFace")
        use_kaggle_checkbox = gr.Checkbox(
            value=kaggle_configured(),
            label="Search Kaggle" + ("" if kaggle_configured() else " (add KAGGLE_USERNAME / KAGGLE_KEY secrets to enable)"),
            interactive=kaggle_configured(),
        )

    search_btn = gr.Button("Search", variant="primary")

    gr.Examples(examples=EXAMPLE_QUERIES, inputs=query_box)

    results_md = gr.Markdown(format_results_markdown([]))

    search_btn.click(
        fn=search_handler,
        inputs=[query_box, top_k_slider, use_hf_checkbox, use_kaggle_checkbox],
        outputs=results_md,
    )
    query_box.submit(
        fn=search_handler,
        inputs=[query_box, top_k_slider, use_hf_checkbox, use_kaggle_checkbox],
        outputs=results_md,
    )

    gr.Markdown(
        """
        ---
        **How ranking works:** results are scored as `0.75 × semantic similarity + 0.25 × quality`.
        Quality blends log-scaled downloads/likes, recency, and metadata completeness — it's a proxy
        for *discoverability and documentation*, not a guarantee of clean data. Always inspect a
        dataset's card before using it in production.
        """
    )

if __name__ == "__main__":
    demo.launch()
