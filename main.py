"""
DS Intern — FastAPI + HTMX web app.

Single global Pipeline instance (no sessions, no concurrency — by design).
Routes:
  GET  /            -> upload form
  POST /upload       -> creates the global pipeline from an uploaded CSV, returns preview
  POST /plan          -> plans a step (e.g. one-hot encode a column), returns a decision card
  POST /commit         -> commits a previously planned step, returns updated preview
  GET  /steps            -> returns the current committed step history (recipe)
"""

from __future__ import annotations

import os
import uuid

from fastapi import FastAPI, Request, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd

from core.pipeline import (
    Pipeline,
    Step,
    OneHotEncodeStep,
    DropNAStep,
    DeduplicateStep,
    RemoveOutliersStep,
    ConvertTypeStep,
    ImputeStep,
    TrimNormalizeStep,
    ParseNumericStep,
    CapOutliersStep,
    LabelEncodeStep,
    DropColumnStep,
    DropSparseColumnsStep,
    RenameColumnStep,
    FilterRowsStep,
)

UPLOAD_DIR = "uploads"
DOWNLOAD_DIR = "downloads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

app = FastAPI(title="DS Intern")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- global state (single-user, single pipeline, by design) ---------------
pipeline: Pipeline | None = None
pending_steps: dict[str, Step] = {}  # step.id -> Step, awaiting commit decision


STEP_REGISTRY = {
    "one_hot_encode": OneHotEncodeStep,
    "drop_na": DropNAStep,
    "deduplicate": DeduplicateStep,
    "remove_outliers": RemoveOutliersStep,
    "convert_type": ConvertTypeStep,
    "impute": ImputeStep,
    "trim_normalize": TrimNormalizeStep,
    "parse_numeric": ParseNumericStep,
    "cap_outliers": CapOutliersStep,
    "label_encode": LabelEncodeStep,
    "drop_column": DropColumnStep,
    "drop_sparse_columns": DropSparseColumnsStep,
    "rename_column": RenameColumnStep,
    "filter_rows": FilterRowsStep,
}
# Step types that don't need a "column" field at all (act on the whole table).
STEPS_WITHOUT_COLUMN = {"deduplicate", "drop_sparse_columns"}
# Steps that need a target_type in addition to a column.
STEPS_NEEDING_TYPE = {"convert_type"}
# Steps that need an imputation strategy (+ optional constant value).
STEPS_NEEDING_STRATEGY = {"impute"}
# Steps that need a case mode.
STEPS_NEEDING_CASE = {"trim_normalize"}


def _columns() -> list[str]:
    if pipeline is None:
        return []
    return pipeline.columns()


def _format_profile(raw_profile: list[dict]) -> list[dict]:
    formatted = []
    for col in raw_profile:
        def fmt(v):
            if v is None:
                return "—"
            try:
                return f"{float(v):.2f}"
            except (TypeError, ValueError):
                return str(v)

        null_pct = col["null_pct"]
        formatted.append({
            "column": col["column"],
            "type": col["type"],
            "min": fmt(col["min"]) if col["mean"] is not None else (col["min"] if col["min"] is not None else "—"),
            "max": fmt(col["max"]) if col["mean"] is not None else (col["max"] if col["max"] is not None else "—"),
            "mean": fmt(col["mean"]),
            "std": fmt(col["std"]),
            "distinct": col["distinct"],
            "null_pct": f"{float(null_pct):.1f}%" if null_pct is not None else "0.0%",
        })
    return formatted


def _preview_context() -> dict:
    if pipeline is None:
        return {"columns": [], "rows": [], "row_count": 0, "applied_steps": [],
                "profile": [], "can_undo": False}
    df = pipeline.preview(50)
    # DuckDB NULLs come back from fetchdf() as NaN in numeric columns, not
    # None. A plain .where(notnull, None) doesn't stick — pandas silently
    # casts None back to NaN on a float64 column — so cast to object dtype
    # first, which can actually hold a real None.
    df = df.astype(object).where(pd.notnull(df), None)
    applied_steps = [
        {"index": i + 1, "description": s.applied_description or s.__class__.__name__}
        for i, s in enumerate(pipeline.steps)
    ]
    return {
        "columns": list(df.columns),
        "rows": df.values.tolist(),
        "row_count": pipeline.row_count(),
        "applied_steps": applied_steps,
        "profile": _format_profile(pipeline.profile()),
        "can_undo": pipeline.can_undo(),
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {**_preview_context(), "columns_for_steps": _columns()},
    )


@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile):
    global pipeline, pending_steps

    filename_lower = file.filename.lower()
    if filename_lower.endswith(".csv"):
        loader = "csv"
    elif filename_lower.endswith((".xlsx", ".xls")):
        loader = "excel"
    else:
        return templates.TemplateResponse(
            request,
            "_upload_error.html",
            {"message": f'"{file.filename}" isn\'t a .csv or .xlsx file. '
                        f"Upload a CSV or Excel file to continue."},
        )

    dest_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{file.filename}")
    with open(dest_path, "wb") as f:
        f.write(await file.read())

    try:
        if loader == "csv":
            new_pipeline = Pipeline.from_csv(dest_path)
        else:
            new_pipeline = Pipeline.from_excel(dest_path)
        # Touch the data now, not lazily, so a malformed file fails here
        # with a clear message instead of surfacing a raw DuckDB traceback
        # the first time the user tries to preview or plan a step.
        new_pipeline.row_count()
    except Exception as e:
        os.remove(dest_path)
        reason = str(e).splitlines()[0] if str(e).strip() else "unreadable file"
        kind = "CSV" if loader == "csv" else "Excel file"
        return templates.TemplateResponse(
            request,
            "_upload_error.html",
            {"message": f"Couldn't read that as a {kind} ({reason})."},
        )

    pipeline = new_pipeline
    pending_steps = {}

    return templates.TemplateResponse(
        request,
        "_workspace.html",
        {**_preview_context(), "columns_for_steps": _columns()},
    )


@app.post("/reset", response_class=HTMLResponse)
def reset(request: Request):
    global pipeline, pending_steps
    pipeline = None
    pending_steps = {}
    return templates.TemplateResponse(
        request,
        "_workspace.html",
        {**_preview_context(), "columns_for_steps": _columns()},
    )


@app.post("/plan", response_class=HTMLResponse)
def plan_step(
    request: Request,
    step_type: str = Form(...),
    column: str = Form(None),
    target_type: str = Form(None),
    strategy: str = Form(None),
    constant_value: str = Form(None),
    case: str = Form(None),
    new_name: str = Form(None),
    threshold_pct: str = Form(None),
    operator: str = Form(None),
    value: str = Form(None),
):
    global pending_steps
    if pipeline is None:
        return HTMLResponse("<div class='error'>Upload a file first.</div>")

    step_cls = STEP_REGISTRY[step_type]
    kwargs = {}
    if column and step_type not in STEPS_WITHOUT_COLUMN:
        kwargs["column"] = column
    if step_type in STEPS_NEEDING_TYPE and target_type:
        kwargs["target_type"] = target_type
    if step_type in STEPS_NEEDING_STRATEGY and strategy:
        kwargs["strategy"] = strategy
        if strategy == "constant":
            kwargs["constant_value"] = constant_value
    if step_type in STEPS_NEEDING_CASE and case:
        kwargs["case"] = case
    if step_type == "rename_column" and new_name:
        kwargs["new_name"] = new_name
    if step_type == "drop_sparse_columns" and threshold_pct:
        try:
            kwargs["threshold_pct"] = float(threshold_pct)
        except ValueError:
            pass
    if step_type == "filter_rows":
        if operator:
            kwargs["operator"] = operator
        if value is not None:
            kwargs["value"] = value

    try:
        step = step_cls(**kwargs)
        result = pipeline.plan(step)
    except (ValueError, TypeError) as e:
        return templates.TemplateResponse(
            request, "_step_error.html", {"message": str(e)}
        )

    pending_steps[step.id] = step

    return templates.TemplateResponse(
        request,
        "_step_card.html",
        {"step_id": step.id, "description": result.description},
    )


@app.post("/commit", response_class=HTMLResponse)
def commit_step(request: Request, step_id: str = Form(...)):
    global pending_steps
    step = pending_steps.get(step_id)
    if pipeline is None or step is None:
        return HTMLResponse("<div class='error'>Nothing pending to commit.</div>")

    try:
        pipeline.commit(step)
    except Exception as e:
        # Something went wrong applying the step (bad column data, an edge
        # case in the generated SQL, etc). The commit form's hx-target is
        # #workspace with outerHTML, so the error response must still be a
        # full workspace (keeping the #workspace id) — otherwise future
        # HTMX calls targeting #workspace would break. Show the error as a
        # banner instead of losing the whole UI, and keep the step pending
        # so nothing is lost.
        return templates.TemplateResponse(
            request,
            "_workspace.html",
            {
                **_preview_context(),
                "columns_for_steps": _columns(),
                "commit_error": f"Couldn't apply that step: {e}",
            },
        )

    pending_steps.pop(step_id, None)

    return templates.TemplateResponse(
        request,
        "_workspace.html",
        {**_preview_context(), "columns_for_steps": _columns()},
    )


@app.post("/undo", response_class=HTMLResponse)
def undo(request: Request):
    if pipeline is not None:
        pipeline.undo()
    return templates.TemplateResponse(
        request,
        "_workspace.html",
        {**_preview_context(), "columns_for_steps": _columns()},
    )


@app.post("/discard", response_class=HTMLResponse)
def discard_step(step_id: str = Form(...)):
    pending_steps.pop(step_id, None)
    return HTMLResponse("")  # step card just disappears


@app.get("/recipe", response_class=HTMLResponse)
def recipe(request: Request):
    if pipeline is None:
        return HTMLResponse("<pre>No pipeline yet.</pre>")
    return templates.TemplateResponse(
        request, "_recipe.html", {"recipe_json": pipeline.export_recipe()}
    )


@app.get("/preview-full", response_class=HTMLResponse)
def preview_full(request: Request):
    """
    Standalone page showing the current pipeline state at a real size —
    not just the 12-row snippet in the main workspace. Meant to be opened
    in a new tab so the person can actually look at the file without
    downloading it.
    """
    if pipeline is None:
        return HTMLResponse("<p style='font-family:monospace;padding:20px;'>"
                             "No dataset loaded yet.</p>")

    limit = 500  # enough to actually look at, still bounded/safe on large data
    df = pipeline.preview(limit)
    df = df.astype(object).where(pd.notnull(df), None)
    total = pipeline.row_count()

    return templates.TemplateResponse(
        request,
        "_preview_full.html",
        {
            "columns": list(df.columns),
            "rows": df.values.tolist(),
            "shown": len(df),
            "total": total,
        },
    )


@app.get("/download/csv")
def download_csv():
    if pipeline is None:
        raise HTTPException(status_code=404, detail="No dataset loaded yet.")
    out_path = os.path.join(DOWNLOAD_DIR, f"{uuid.uuid4().hex}_cleaned.csv")
    pipeline.export_csv(out_path)
    return FileResponse(out_path, media_type="text/csv", filename="cleaned.csv")


@app.get("/download/parquet")
def download_parquet():
    if pipeline is None:
        raise HTTPException(status_code=404, detail="No dataset loaded yet.")
    out_path = os.path.join(DOWNLOAD_DIR, f"{uuid.uuid4().hex}_cleaned.parquet")
    pipeline.export_parquet(out_path)
    return FileResponse(out_path, media_type="application/octet-stream", filename="cleaned.parquet")
