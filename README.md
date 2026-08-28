# DS Intern

An interactive, web-based data cleaning pipeline. Upload a CSV or Excel
file, propose cleaning steps one at a time, see exactly what each step
will do before it runs, and download the cleaned result as CSV or Parquet
(or preview it full-size in a new tab without downloading).

## Run it

    pip install -r requirements.txt
    python -m uvicorn main:app --reload

Then open http://127.0.0.1:8000

(Use `python -m uvicorn ...` rather than bare `uvicorn ...` — on Windows,
pip-installed scripts often aren't on PATH even after a successful install.)

## If the browser shows stale data after restarting the server

Pipeline state lives only in memory in the running process — there's no
database. If you stop the server and old data still shows up after
restart, the old process is very likely still running in the background
(Windows sometimes keeps a process alive after its terminal window closes).
Check for it before starting a new one:

    netstat -ano | findstr :8000
    taskkill /PID <pid_from_above> /F

## Architecture

- `core/pipeline.py` — the engine. `Pipeline` holds a DuckDB connection
  and the chain of applied steps; each `Step` subclass implements
  `.plan()` (compute what would happen, without applying it) and gets
  applied via `.commit()`. Nothing is UI-specific here.
- `main.py` — FastAPI + HTMX web layer. A single global `Pipeline`
  instance (no sessions — single-user by design). Routes: `/upload`,
  `/plan`, `/commit`, `/undo`, `/reset`, `/recipe`, `/preview-full`,
  `/download/csv`, `/download/parquet`.
- `templates/` — Jinja + HTMX fragments, swapped in-place without full
  page reloads.
- `static/style.css` — design tokens (flat, utilitarian, monospace for
  data — no purple, no gradients).

## Step catalog (14 step types)

- **Missing values**: drop rows with nulls, fill nulls (mean / median /
  mode / constant), drop columns above a null-% threshold
- **Duplicates**: remove exact duplicate rows
- **Outliers**: remove (IQR fence) or cap/winsorize
- **Text cleanup**: trim whitespace + normalize case, parse numeric
  strings (strips currency symbols/commas)
- **Type & encoding**: convert type (int/float/text/date/boolean),
  one-hot encode, label-encode
- **Columns**: rename, drop
- **Rows**: filter by condition (==, !=, >, >=, <, <=, contains)

Every step shows a computed preview description (e.g. "Fill 4 missing
values in 'age' with the median (30.25)") before you confirm it, and
every committed step can be undone one at a time.

## Viewing your data

- The main page shows a 12-row inline preview after every step.
- Click "Preview full ↗" to open a dedicated tab showing up to 500 rows
  of the *current* pipeline state (after whatever steps you've applied
  so far) — no download needed, updates automatically as you commit
  more steps and reopen it.
- "Download CSV" / "Download Parquet" give you the complete file
  (all rows) to save locally.

## Design decisions worth knowing

- Engine is DuckDB + views chained on top of each other — nothing is
  materialized into Python except bounded previews (LIMIT-based) and
  the column profile (`SUMMARIZE`, one pass, safe at scale).
- CSV uses DuckDB's native `read_csv_auto` (lazy/streaming). Excel is
  loaded via pandas/openpyxl since `.xlsx` is inherently capped around
  ~1M rows/sheet — no out-of-core story needed there.
- Single global pipeline, no concurrency/session handling — a
  deliberate scope decision, not an oversight.
- Target dataset scale: comfortably up to ~2GB CSV files.
- If `/commit` fails partway (bad data, an edge case in generated SQL),
  the error shows as a banner in the UI instead of a raw JSON/500 —
  the pending step is preserved and the app stays usable.

## Known gaps (not yet built)

- Date parsing with explicit format strings (auto-detect struggles on
  ambiguous formats like 01/02/2024)
- Fuzzy/near-duplicate detection
- Split/merge column operations
- Scaling/normalization and target encoding — intentionally held for
  phase 2 (model training), since they're meaningless without that
  context
