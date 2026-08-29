# DS Intern
 
It is an interactive, web-based data cleaning and EDA pipeline with a terminal-style UI. Upload a CSV or Excel file, propose cleaning steps one at a time, see exactly what each step will do before it runs, explore the data with auto-generated charts, and download the cleaned result as CSV or Parquet.
 
## Run it
 
    pip install -r requirements.txt
    python -m uvicorn main:app --reload
 
Then open http://127.0.0.1:8000
 
(Use `python -m uvicorn ...` rather than bare `uvicorn ...` — on Windows, pip-installed scripts often aren't on PATH even after a successful install.)
 
## If the browser shows stale data after restarting the server
 
Pipeline state lives only in memory in the running process — there's no database. If you stop the server and old data still shows up after restart, an old process is very likely still running in the background (Windows sometimes keeps a process alive after its terminal window closes). Check for it before starting a new one:
 
    netstat -ano | findstr :8000
    taskkill /PID <pid_from_above> /F
 
## Architecture
 
- `core/pipeline.py` — the engine. `Pipeline` holds a DuckDB connection and the chain of applied steps; each `Step` subclass implements `.plan()` (compute what would happen, without applying it) and gets applied via `.commit()`. Nothing here is UI-specific — the same engine drives every route.
- `main.py` — FastAPI + HTMX web layer. A single global `Pipeline` instance (no sessions, no concurrency — single-user by design). Routes:
  - `POST /upload` — load a CSV or Excel file
  - `POST /plan` / `POST /commit` / `POST /discard` — propose, apply, or skip a cleaning step
  - `POST /undo` — roll back the most recently committed step
  - `POST /reset` — clear the current dataset
  - `GET /recipe` — JSON log of committed steps (reproducible)
  - `GET /preview-full` — full-size data preview in a new tab (up to 500 rows), no download needed
  - `GET /eda` — histograms, category bar charts, and a correlation matrix for the current data
  - `GET /download/csv`, `GET /download/parquet` — export the cleaned result
- `templates/` — Jinja + HTMX fragments, swapped in place without full page reloads.
- `static/style.css` — terminal aesthetic: black background, white borders, monospace throughout, sharp corners, green/amber/red accents for confirm/pending/error states.
## Step catalog (14 step types)
 
- **Missing values**: drop rows with nulls, fill nulls (mean / median / mode / constant), drop columns above a null-% threshold
- **Duplicates**: remove exact duplicate rows
- **Outliers**: remove (IQR fence) or cap/winsorize
- **Text cleanup**: trim whitespace + normalize case, parse numeric strings (strips currency symbols/commas)
- **Type & encoding**: convert type (int/float/text/date/boolean), one-hot encode, label-encode
- **Columns**: rename, drop
- **Rows**: filter by condition (==, !=, >, >=, <, <=, contains)
Every step shows a computed preview description (e.g. "Fill 4 missing values in 'age' with the median (30.25)") before you confirm it, and every committed step can be undone one at a time.
 
## EDA
 
Opens in a new tab (`EDA ↗` next to the download buttons) and always reflects the *current* pipeline state — i.e. after whatever cleaning steps you've applied so far, not the original upload:
 
- **Histograms** for every numeric column (equal-width binning, computed in a single bounded DuckDB query)
- **Bar charts** of value counts for low-cardinality categorical columns (capped at 30 distinct values — a free-text column won't get a nonsensical chart)
- **Correlation matrix** (Pearson) across numeric columns, capped at 8 columns for legibility
Charts render client-side via Chart.js (loaded from CDN); all aggregation happens server-side in DuckDB, so nothing is pulled row-by-row into Python.
 
## Viewing your data
 
- The main page shows a 12-row inline preview after every step.
- **Preview full ↗** opens a dedicated tab showing up to 500 rows of the current state — no download needed.
- **EDA ↗** opens the charts/correlation view described above.
- **Download CSV** / **Download Parquet** give you the complete file (all rows) to save locally.
## Design decisions 

- Engine is DuckDB + views chained on top of each other — nothing is materialized into Python except bounded previews (LIMIT-based), the column profile (`SUMMARIZE`, one pass), and EDA aggregates (histograms/value-counts/correlations, all computed via grouped/aggregate SQL, not row iteration).
- CSV uses DuckDB's native `read_csv_auto` (lazy/streaming). Excel is loaded via pandas/openpyxl since `.xlsx` is inherently capped around ~1M rows/sheet — no out-of-core story needed there.
- Single global pipeline, no concurrency/session handling — a deliberate scope decision, not an oversight.
- Target dataset scale: comfortably up to ~2GB CSV files.
- If `/commit` fails partway (bad data, an edge case in generated SQL), the error shows as a banner in the UI instead of a raw JSON/500 — the pending step is preserved and the app stays usable.
## Goals for version 2
 
- Date parsing with explicit format strings (auto-detect struggles on ambiguous formats like 01/02/2024)
- Fuzzy/near-duplicate detection
- Split/merge column operations
- Scaling/normalization and target encoding — intentionally held for phase 2 (model training), since they're meaningless without that context
