# Speech2Market

Speech2Market predicts short-term directional moves in **SPX, GOLD, VIX, and TNX** from Federal Reserve speech content combined with market and macro context. Fed speeches are scraped, embedded with FinBERT, and fed into a two-stage sign + magnitude model stack that forecasts moves at **t+3, t+7, and t+30** day horizons.

A live demo of the pipeline's predictions is deployed as a Streamlit app ("Speech2Market Live Demo").

## How it works

1. **Scrape** — Fed speeches/testimony are pulled and stored (deduplicated by link) in a Supabase Postgres table (`fed_speech`), alongside `price_action` and `macro_indicators` tables synced on a schedule.
2. **Embed** — Speech text is chunked (512-token windows, 50 stride) and embedded with an ONNX FinBERT model, mean-pooled across chunks into a single vector per speech.
3. **Model** — `EL_main.ipynb` trains the full pipeline: for each of the 12 targets (4 assets × 3 horizons), a **sign classifier** (`HistGradientBoostingClassifier`, Optuna-tuned, OOF-swept decision threshold) predicts direction, and a **magnitude regressor** (`HistGradientBoostingRegressor`) predicts move size. Final prediction = sign × magnitude. Validation uses purged `TimeSeriesSplit` to prevent lookahead leakage, and bootstrap significance testing checks whether the FinBERT embedding features actually add value over a no-embedding baseline.
4. **Serve** — Trained artifacts (sign models, magnitude models, per-target thresholds, feature columns) are loaded by the Streamlit app to generate live predictions with an associated up-probability.

## Repo structure

| Path | Purpose |
|---|---|
| `EL_main.ipynb` | Main modeling pipeline — trains both the sign and magnitude classifiers for all targets, runs Optuna tuning, CV, and significance testing, and saves production artifacts. |
| `models_el/production/` | Saved model artifacts: sign models, magnitude models, decision thresholds, feature columns, metadata. |
| `fed_speech_embeddings.npy` | Cached FinBERT embeddings for scraped speeches. |
| `TOOLS_automate_scrape_db.py` | Automated Fed speech scraper — inserts new speeches into Supabase, run on a daily schedule. |
| `TOOLS_fed_scrapper.ipynb` | Notebook version of the speech scraper (development/backfill). |
| `TOOLS_backfill_scraper.ipynb` | Backfills historical Fed speeches into the database. |
| `TOOLS_get_newspeeches.ipynb` | Fetches newly published speeches. |
| `TOOLS_embed.ipynb` | Embeds speech text into vectors using the ONNX FinBERT model. |
| `TOOLS_copy_npy_to_db.ipynb` | Copies locally cached `.npy` embeddings into the Supabase embedding column. |
| `TOOLS_get_macros.ipynb` | Pulls macroeconomic indicators (FRED) into the database. |
| `TOOLS_get_prices.ipynb` | Pulls asset price data (yfinance) into the database. |
| `TOOLS_merge_fed_speech.ipynb` | Merges/reconciles speech records (e.g. dedup, source merges). |
| `TOOLS_fix_parsing.ipynb` | One-off fixes for text-parsing issues in scraped speeches. |
| `TOOLS_feature_importance_check.ipynb` | Inspects feature importance for trained models. |
| `requirements.txt` | Python dependencies. |

## Setup

```bash
pip install -r requirements.txt
```

Configure Supabase credentials (and any other required API keys) as environment variables before running the scraper, embedder, or training notebook.

## Running the pipeline

1. Scrape speeches: `TOOLS_automate_scrape_db.py` (or the notebook equivalents for backfill).
2. Embed speeches: `TOOLS_embed.ipynb`.
3. Pull supporting market/macro data: `TOOLS_get_prices.ipynb`, `TOOLS_get_macros.ipynb`.
4. Train models: `EL_main.ipynb` — trains and saves the sign + magnitude classifiers to `models_el/production/`.
5. Serve predictions via the Streamlit app (see live demo).

## Notes

- CV is purged (`TimeSeriesSplit` with `train_idx[:-horizon]`) to avoid leakage across the label horizon.
- Embedding value is validated per target via bootstrap significance testing (2000 resamples) against a no-embedding baseline, rather than assumed.
