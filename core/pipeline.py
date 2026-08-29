"""
Core Pipeline/Step engine for DS Intern.
 
Design goals:
- CLI and web both drive this same engine — it has no knowledge of how
  it's being presented to the user.
- Every transform is expressed as SQL run through DuckDB, so nothing
  is ever fully materialized into Python memory unless explicitly asked
  for (preview only).
- Steps are inspectable *before* they run, so a UI (CLI prompt or web
  API) can show "here's what I'm about to do" and get a decision back.
- The whole pipeline can be serialized to a JSON "recipe" so it's
  reproducible against a new file later.
"""
 
from __future__ import annotations
 
import json
import uuid
from dataclasses import dataclass, field
from typing import Optional
 
import duckdb
 
 
# ---------------------------------------------------------------------------
# Step abstraction
# ---------------------------------------------------------------------------
 
@dataclass
class StepResult:
    """What a Step reports about itself before it's applied."""
    description: str          # human-readable, shown in the "should I do this?" prompt
    sql: str                  # the SQL that will run if approved
    preview_sql: str          # a LIMIT-bounded version of sql, safe to run eagerly
 
 
class Step:
    """
    Base class for a single pipeline operation.
 
    Subclasses implement `plan(pipeline)`, which returns a StepResult
    describing what SQL would run against the pipeline's *current* view,
    without executing it. The pipeline (or a UI layer) decides whether
    to call `.commit()`.
    """
 
    def __init__(self, **params):
        self.id = str(uuid.uuid4())[:8]
        self.params = params
        self._planned: Optional[StepResult] = None
        self.applied_description: Optional[str] = None
 
    # Subclasses override this.
    def plan(self, pipeline: "Pipeline") -> StepResult:
        raise NotImplementedError
 
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.__class__.__name__,
            "params": self.params,
        }
 
 
class OneHotEncodeStep(Step):
    """One-hot encode a low-cardinality categorical column."""
 
    def __init__(self, column: str, max_categories: int = 20):
        super().__init__(column=column, max_categories=max_categories)
 
    def plan(self, pipeline: "Pipeline") -> StepResult:
        column = self.params["column"]
        con = pipeline.con
        current_view = pipeline.current_view
 
        # Discover distinct values (bounded query — safe even on huge tables,
        # since DISTINCT + LIMIT lets DuckDB short-circuit once it has enough).
        distinct = con.execute(
            f'SELECT DISTINCT "{column}" FROM {current_view} '
            f'WHERE "{column}" IS NOT NULL LIMIT {self.params["max_categories"] + 1}'
        ).fetchall()
        values = [row[0] for row in distinct]
 
        if len(values) > self.params["max_categories"]:
            raise ValueError(
                f'"{column}" has more than {self.params["max_categories"]} '
                f"distinct values — one-hot encoding not recommended here. "
                f"Consider label/target encoding instead."
            )
 
        case_columns = ",\n    ".join(
            f'CASE WHEN "{column}" = \'{v}\' THEN 1 ELSE 0 END AS "{column}_{v}"'
            for v in values
        )
        sql = f'SELECT *, \n    {case_columns}\nFROM {current_view}'
        preview_sql = sql + " LIMIT 50"
 
        return StepResult(
            description=(
                f'One-hot encode "{column}" '
                f"({len(values)} categories: {', '.join(map(str, values))}) "
                f"→ adds {len(values)} new binary columns."
            ),
            sql=sql,
            preview_sql=preview_sql,
        )
 
 
class LabelEncodeStep(Step):
    """
    Map each distinct category to an integer code (0, 1, 2, ...), assigned
    in sorted order for reproducibility. Good for high-cardinality columns
    where one-hot would blow up the column count, or for genuinely ordinal
    data. Replaces the column in place rather than adding new columns.
    """
 
    def __init__(self, column: str, max_categories: int = 1000):
        super().__init__(column=column, max_categories=max_categories)
 
    def plan(self, pipeline: "Pipeline") -> StepResult:
        column = self.params["column"]
        con = pipeline.con
        current_view = pipeline.current_view
 
        distinct = con.execute(
            f'SELECT DISTINCT "{column}" FROM {current_view} '
            f'WHERE "{column}" IS NOT NULL '
            f'ORDER BY "{column}" '
            f'LIMIT {self.params["max_categories"] + 1}'
        ).fetchall()
        values = [row[0] for row in distinct]
 
        if len(values) > self.params["max_categories"]:
            raise ValueError(
                f'"{column}" has more than {self.params["max_categories"]} '
                f"distinct values — that's a lot to label-encode at once. "
                f"Double check this is the right column."
            )
 
        case_expr = "\n    ".join(
            f"WHEN '{v}' THEN {i}" for i, v in enumerate(values)
        )
        other_cols = [c for c in pipeline.columns() if c != column]
        select_cols = ", ".join(f'"{c}"' for c in other_cols)
        select_cols = (select_cols + ", ") if select_cols else ""
 
        sql = (
            f'SELECT {select_cols}CASE "{column}" \n    {case_expr}\n    '
            f'ELSE NULL END AS "{column}" FROM {current_view}'
        )
        preview_sql = sql + " LIMIT 50"
 
        mapping_preview = ", ".join(f"{v}→{i}" for i, v in enumerate(values[:6]))
        if len(values) > 6:
            mapping_preview += f", …+{len(values) - 6} more"
 
        return StepResult(
            description=(
                f'Label-encode "{column}" ({len(values)} categories) as integers: '
                f"{mapping_preview}."
            ),
            sql=sql,
            preview_sql=preview_sql,
        )
 
 
class DropNAStep(Step):
    """Drop rows where a given column is NULL."""
 
    def __init__(self, column: str):
        super().__init__(column=column)
 
    def plan(self, pipeline: "Pipeline") -> StepResult:
        column = self.params["column"]
        current_view = pipeline.current_view
        con = pipeline.con
 
        null_count = con.execute(
            f'SELECT COUNT(*) FROM {current_view} WHERE "{column}" IS NULL'
        ).fetchone()[0]
 
        sql = f'SELECT * FROM {current_view} WHERE "{column}" IS NOT NULL'
        preview_sql = sql + " LIMIT 50"
 
        return StepResult(
            description=f'Drop {null_count} rows where "{column}" is NULL.',
            sql=sql,
            preview_sql=preview_sql,
        )
 
 
class ImputeStep(Step):
    """
    Fill NULLs in a column instead of dropping the row. Alternative to
    DropNAStep — mean/median/mode for numeric-ish gaps, or a fixed
    constant value (works for any type, including text columns).
    """
 
    STRATEGIES = {"mean", "median", "mode", "constant"}
 
    def __init__(self, column: str, strategy: str, constant_value: str = None):
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f'Unknown imputation strategy "{strategy}". '
                f"Choose one of: {', '.join(self.STRATEGIES)}."
            )
        if strategy == "constant" and constant_value is None:
            raise ValueError('"constant" strategy needs a value to fill with.')
        super().__init__(column=column, strategy=strategy, constant_value=constant_value)
 
    def plan(self, pipeline: "Pipeline") -> StepResult:
        column = self.params["column"]
        strategy = self.params["strategy"]
        current_view = pipeline.current_view
        con = pipeline.con
 
        null_count = con.execute(
            f'SELECT COUNT(*) FROM {current_view} WHERE "{column}" IS NULL'
        ).fetchone()[0]
 
        if null_count == 0:
            raise ValueError(f'"{column}" has no missing values to fill.')
 
        other_cols = [c for c in pipeline.columns() if c != column]
        select_cols = ", ".join(f'"{c}"' for c in other_cols)
        select_cols = (select_cols + ", ") if select_cols else ""
 
        if strategy == "mean":
            fill_expr = f'(SELECT AVG("{column}") FROM {current_view})'
            fill_value = con.execute(f"SELECT {fill_expr}").fetchone()[0]
            label = f"mean ({fill_value:.2f})" if fill_value is not None else "mean"
        elif strategy == "median":
            fill_expr = f'(SELECT median("{column}") FROM {current_view})'
            fill_value = con.execute(f"SELECT {fill_expr}").fetchone()[0]
            label = f"median ({fill_value:.2f})" if fill_value is not None else "median"
        elif strategy == "mode":
            fill_expr = f'(SELECT mode("{column}") FROM {current_view})'
            fill_value = con.execute(f"SELECT {fill_expr}").fetchone()[0]
            label = f"mode ({fill_value})" if fill_value is not None else "mode"
        else:  # constant
            raw = self.params["constant_value"]
            fill_expr = f"'{raw}'"
            label = f'constant value "{raw}"'
 
        sql = (
            f'SELECT {select_cols}COALESCE("{column}", {fill_expr}) AS "{column}" '
            f"FROM {current_view}"
        )
        preview_sql = sql + " LIMIT 50"
 
        return StepResult(
            description=f'Fill {null_count} missing values in "{column}" with the {label}.',
            sql=sql,
            preview_sql=preview_sql,
        )
 
 
class TrimNormalizeStep(Step):
    """Trim whitespace and normalize casing on a text column."""
 
    CASE_MODES = {"none", "lower", "upper", "title"}
 
    def __init__(self, column: str, case: str = "none"):
        if case not in self.CASE_MODES:
            raise ValueError(f'Unknown case mode "{case}". Choose one of: {", ".join(self.CASE_MODES)}.')
        super().__init__(column=column, case=case)
 
    def plan(self, pipeline: "Pipeline") -> StepResult:
        column = self.params["column"]
        case = self.params["case"]
        current_view = pipeline.current_view
        con = pipeline.con
 
        affected = con.execute(
            f'SELECT COUNT(*) FROM {current_view} '
            f'WHERE "{column}" != TRIM("{column}")'
        ).fetchone()[0]
 
        expr = f'TRIM("{column}")'
        case_label = ""
        if case == "lower":
            expr = f"LOWER({expr})"
            case_label = ", lowercased"
        elif case == "upper":
            expr = f"UPPER({expr})"
            case_label = ", uppercased"
        elif case == "title":
            expr = f"array_to_string(list_transform(string_split({expr}, ' '), x -> upper(x[1:1]) || lower(x[2:])), ' ')"
            case_label = ", title-cased"
 
        other_cols = [c for c in pipeline.columns() if c != column]
        select_cols = ", ".join(f'"{c}"' for c in other_cols)
        select_cols = (select_cols + ", ") if select_cols else ""
 
        sql = f'SELECT {select_cols}{expr} AS "{column}" FROM {current_view}'
        preview_sql = sql + " LIMIT 50"
 
        return StepResult(
            description=f'Trim whitespace{case_label} on "{column}" ({affected} rows affected by trimming).',
            sql=sql,
            preview_sql=preview_sql,
        )
 
 
class ParseNumericStep(Step):
    """Strip currency symbols/commas/whitespace from a text column and cast to DOUBLE."""
 
    def __init__(self, column: str):
        super().__init__(column=column)
 
    def plan(self, pipeline: "Pipeline") -> StepResult:
        column = self.params["column"]
        current_view = pipeline.current_view
        con = pipeline.con
 
        cleaned_expr = f'''regexp_replace("{column}", '[^0-9.\\-]', '', 'g')'''
        would_fail = con.execute(
            f'SELECT COUNT(*) FROM {current_view} '
            f'WHERE "{column}" IS NOT NULL '
            f"AND TRY_CAST({cleaned_expr} AS DOUBLE) IS NULL"
        ).fetchone()[0]
 
        other_cols = [c for c in pipeline.columns() if c != column]
        select_cols = ", ".join(f'"{c}"' for c in other_cols)
        select_cols = (select_cols + ", ") if select_cols else ""
 
        sql = (
            f'SELECT {select_cols}TRY_CAST({cleaned_expr} AS DOUBLE) AS "{column}" '
            f"FROM {current_view}"
        )
        preview_sql = sql + " LIMIT 50"
 
        warning = f" ({would_fail} values can't parse and will become NULL)" if would_fail else ""
        return StepResult(
            description=f'Parse "{column}" as a number, stripping symbols like $ and , {warning}.',
            sql=sql,
            preview_sql=preview_sql,
        )
 
 
class DeduplicateStep(Step):
    """Drop exact duplicate rows."""
 
    def plan(self, pipeline: "Pipeline") -> StepResult:
        current_view = pipeline.current_view
        con = pipeline.con
 
        total = con.execute(f"SELECT COUNT(*) FROM {current_view}").fetchone()[0]
        distinct = con.execute(
            f"SELECT COUNT(*) FROM (SELECT DISTINCT * FROM {current_view})"
        ).fetchone()[0]
        dupes = total - distinct
 
        sql = f"SELECT DISTINCT * FROM {current_view}"
        preview_sql = sql + " LIMIT 50"
 
        return StepResult(
            description=f"Remove {dupes} exact duplicate rows (of {total} total).",
            sql=sql,
            preview_sql=preview_sql,
        )
 
 
class RemoveOutliersStep(Step):
    """
    Drop rows where a numeric column falls outside the IQR fence
    (Q1 - k*IQR, Q3 + k*IQR). Default k=1.5 is the standard Tukey fence.
    """
 
    def __init__(self, column: str, k: float = 1.5):
        super().__init__(column=column, k=k)
 
    def plan(self, pipeline: "Pipeline") -> StepResult:
        column = self.params["column"]
        k = self.params["k"]
        current_view = pipeline.current_view
        con = pipeline.con
 
        q1, q3 = con.execute(
            f'SELECT quantile_cont("{column}", 0.25), quantile_cont("{column}", 0.75) '
            f"FROM {current_view}"
        ).fetchone()
 
        if q1 is None or q3 is None:
            raise ValueError(
                f'"{column}" has no usable numeric values to compute quartiles from.'
            )
 
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
 
        outlier_count = con.execute(
            f'SELECT COUNT(*) FROM {current_view} '
            f'WHERE "{column}" < {lower} OR "{column}" > {upper}'
        ).fetchone()[0]
 
        sql = (
            f'SELECT * FROM {current_view} '
            f'WHERE "{column}" BETWEEN {lower} AND {upper} '
            f'OR "{column}" IS NULL'
        )
        preview_sql = sql + " LIMIT 50"
 
        return StepResult(
            description=(
                f'Remove {outlier_count} outlier rows in "{column}" '
                f"outside [{lower:.2f}, {upper:.2f}] (IQR fence, k={k})."
            ),
            sql=sql,
            preview_sql=preview_sql,
        )
 
 
class CapOutliersStep(Step):
    """
    Winsorize: clip values outside the IQR fence to the fence bounds
    instead of dropping the row. Preferred over RemoveOutliersStep when
    you want to keep every row (e.g. the other columns' data is still useful).
    """
 
    def __init__(self, column: str, k: float = 1.5):
        super().__init__(column=column, k=k)
 
    def plan(self, pipeline: "Pipeline") -> StepResult:
        column = self.params["column"]
        k = self.params["k"]
        current_view = pipeline.current_view
        con = pipeline.con
 
        q1, q3 = con.execute(
            f'SELECT quantile_cont("{column}", 0.25), quantile_cont("{column}", 0.75) '
            f"FROM {current_view}"
        ).fetchone()
 
        if q1 is None or q3 is None:
            raise ValueError(f'"{column}" has no usable numeric values to compute quartiles from.')
 
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
 
        affected = con.execute(
            f'SELECT COUNT(*) FROM {current_view} '
            f'WHERE "{column}" < {lower} OR "{column}" > {upper}'
        ).fetchone()[0]
 
        other_cols = [c for c in pipeline.columns() if c != column]
        select_cols = ", ".join(f'"{c}"' for c in other_cols)
        select_cols = (select_cols + ", ") if select_cols else ""
 
        clip_expr = (
            f'CASE WHEN "{column}" < {lower} THEN {lower} '
            f'WHEN "{column}" > {upper} THEN {upper} '
            f'ELSE "{column}" END'
        )
        sql = f'SELECT {select_cols}{clip_expr} AS "{column}" FROM {current_view}'
        preview_sql = sql + " LIMIT 50"
 
        return StepResult(
            description=(
                f'Cap {affected} outlier values in "{column}" to '
                f"[{lower:.2f}, {upper:.2f}] (IQR fence, k={k}) instead of dropping rows."
            ),
            sql=sql,
            preview_sql=preview_sql,
        )
 
 
class ConvertTypeStep(Step):
    """Cast a column to a target SQL type."""
 
    ALLOWED_TYPES = {
        "integer": "BIGINT",
        "float": "DOUBLE",
        "text": "VARCHAR",
        "date": "DATE",
        "boolean": "BOOLEAN",
    }
 
    def __init__(self, column: str, target_type: str):
        if target_type not in self.ALLOWED_TYPES:
            raise ValueError(
                f'Unsupported target type "{target_type}". '
                f"Choose one of: {', '.join(self.ALLOWED_TYPES)}."
            )
        super().__init__(column=column, target_type=target_type)
 
    def plan(self, pipeline: "Pipeline") -> StepResult:
        column = self.params["column"]
        target_type = self.params["target_type"]
        sql_type = self.ALLOWED_TYPES[target_type]
        current_view = pipeline.current_view
        con = pipeline.con
 
        # Count how many non-null values would fail the cast, so the user
        # knows up front if this is going to quietly turn values into NULL.
        would_fail = con.execute(
            f'SELECT COUNT(*) FROM {current_view} '
            f'WHERE "{column}" IS NOT NULL '
            f'AND TRY_CAST("{column}" AS {sql_type}) IS NULL'
        ).fetchone()[0]
 
        other_cols = [c for c in pipeline.columns() if c != column]
        select_cols = ", ".join(f'"{c}"' for c in other_cols)
        select_cols = (select_cols + ", ") if select_cols else ""
 
        sql = (
            f'SELECT {select_cols}TRY_CAST("{column}" AS {sql_type}) AS "{column}" '
            f"FROM {current_view}"
        )
        preview_sql = sql + " LIMIT 50"
 
        warning = (
            f" ({would_fail} values can't convert and will become NULL)"
            if would_fail else ""
        )
        return StepResult(
            description=f'Convert "{column}" to {target_type}{warning}.',
            sql=sql,
            preview_sql=preview_sql,
        )
 
 
class DropColumnStep(Step):
    """Drop one column entirely."""
 
    def __init__(self, column: str):
        super().__init__(column=column)
 
    def plan(self, pipeline: "Pipeline") -> StepResult:
        column = self.params["column"]
        current_view = pipeline.current_view
        all_cols = pipeline.columns()
 
        if column not in all_cols:
            raise ValueError(f'No column named "{column}".')
        if len(all_cols) == 1:
            raise ValueError("Can't drop the only remaining column.")
 
        keep_cols = [c for c in all_cols if c != column]
        select_cols = ", ".join(f'"{c}"' for c in keep_cols)
 
        sql = f"SELECT {select_cols} FROM {current_view}"
        preview_sql = sql + " LIMIT 50"
 
        return StepResult(
            description=f'Drop column "{column}".',
            sql=sql,
            preview_sql=preview_sql,
        )
 
 
class DropSparseColumnsStep(Step):
    """
    Drop any column whose null percentage is at or above a threshold —
    different from dropping rows: sometimes the column itself carries too
    little signal to be worth keeping, regardless of which rows it hits.
    """
 
    def __init__(self, threshold_pct: float = 50.0):
        super().__init__(threshold_pct=threshold_pct)
 
    def plan(self, pipeline: "Pipeline") -> StepResult:
        threshold = self.params["threshold_pct"]
        current_view = pipeline.current_view
        con = pipeline.con
        all_cols = pipeline.columns()
 
        total = con.execute(f"SELECT COUNT(*) FROM {current_view}").fetchone()[0]
        if total == 0:
            raise ValueError("No rows to evaluate null percentage against.")
 
        sparse_cols = []
        for col in all_cols:
            nulls = con.execute(
                f'SELECT COUNT(*) FROM {current_view} WHERE "{col}" IS NULL'
            ).fetchone()[0]
            if (nulls / total) * 100 >= threshold:
                sparse_cols.append(col)
 
        if not sparse_cols:
            raise ValueError(f"No columns are {threshold}% or more null — nothing to drop.")
 
        keep_cols = [c for c in all_cols if c not in sparse_cols]
        if not keep_cols:
            raise ValueError(
                f"All columns are {threshold}% or more null — dropping them all "
                f"would leave nothing. Lower the threshold."
            )
 
        select_cols = ", ".join(f'"{c}"' for c in keep_cols)
        sql = f"SELECT {select_cols} FROM {current_view}"
        preview_sql = sql + " LIMIT 50"
 
        return StepResult(
            description=(
                f'Drop {len(sparse_cols)} column(s) that are ≥{threshold}% null: '
                f"{', '.join(sparse_cols)}."
            ),
            sql=sql,
            preview_sql=preview_sql,
        )
 
 
class RenameColumnStep(Step):
    """Rename one column."""
 
    def __init__(self, column: str, new_name: str):
        if not new_name or not new_name.strip():
            raise ValueError("New column name can't be empty.")
        super().__init__(column=column, new_name=new_name.strip())
 
    def plan(self, pipeline: "Pipeline") -> StepResult:
        column = self.params["column"]
        new_name = self.params["new_name"]
        current_view = pipeline.current_view
        all_cols = pipeline.columns()
 
        if column not in all_cols:
            raise ValueError(f'No column named "{column}".')
        if new_name in all_cols and new_name != column:
            raise ValueError(f'A column named "{new_name}" already exists.')
 
        select_parts = [
            f'"{c}" AS "{new_name}"' if c == column else f'"{c}"'
            for c in all_cols
        ]
        sql = f"SELECT {', '.join(select_parts)} FROM {current_view}"
        preview_sql = sql + " LIMIT 50"
 
        return StepResult(
            description=f'Rename column "{column}" to "{new_name}".',
            sql=sql,
            preview_sql=preview_sql,
        )
 
 
class FilterRowsStep(Step):
    """
    Keep only rows matching a simple column/operator/value condition.
    General-purpose row filter — not tied to outliers specifically.
    """
 
    OPERATORS = {
        "==": "=", "!=": "!=", ">": ">", ">=": ">=", "<": "<", "<=": "<=",
        "contains": "LIKE",
    }
 
    def __init__(self, column: str, operator: str, value: str):
        if operator not in self.OPERATORS:
            raise ValueError(
                f'Unknown operator "{operator}". Choose one of: {", ".join(self.OPERATORS)}.'
            )
        super().__init__(column=column, operator=operator, value=value)
 
    def plan(self, pipeline: "Pipeline") -> StepResult:
        column = self.params["column"]
        operator = self.params["operator"]
        value = self.params["value"]
        current_view = pipeline.current_view
        con = pipeline.con
 
        sql_op = self.OPERATORS[operator]
        if operator == "contains":
            condition = f'"{column}" LIKE \'%{value}%\''
        elif operator in ("==", "!="):
            # Try numeric comparison first, fall back to string equality.
            condition = f'CAST("{column}" AS VARCHAR) {sql_op} \'{value}\''
        else:
            condition = f'TRY_CAST("{column}" AS DOUBLE) {sql_op} TRY_CAST(\'{value}\' AS DOUBLE)'
 
        total = con.execute(f"SELECT COUNT(*) FROM {current_view}").fetchone()[0]
        matching = con.execute(
            f"SELECT COUNT(*) FROM {current_view} WHERE {condition}"
        ).fetchone()[0]
 
        sql = f"SELECT * FROM {current_view} WHERE {condition}"
        preview_sql = sql + " LIMIT 50"
 
        return StepResult(
            description=(
                f'Keep rows where "{column}" {operator} "{value}" '
                f"({matching} of {total} rows match; {total - matching} will be removed)."
            ),
            sql=sql,
            preview_sql=preview_sql,
        )
 
 
# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
 
class Pipeline:
    """
    Holds one DuckDB connection, the chain of committed steps, and the
    name of the current view representing "the data after all committed
    steps so far".
 
    Usage pattern (same for CLI or web):
 
        pipeline = Pipeline.from_csv("data.csv")
        step = OneHotEncodeStep(column="city")
        result = pipeline.plan(step)      # -> StepResult, show to user
        # ... user says yes ...
        pipeline.commit(step)             # applies it, advances current_view
        pipeline.preview()                # cheap LIMIT-bounded preview
 
    Nothing here ever pulls the full table into Python — `preview()` and
    `plan()` are the only places data crosses into Python, and both are
    LIMIT-bounded.
    """
 
    def __init__(self, con: duckdb.DuckDBPyConnection, base_view: str):
        self.con = con
        self.current_view = base_view
        self.steps: list[Step] = []
        self._view_counter = 0
        # Parallel to self.steps but one longer — view_history[0] is the
        # base view before any steps, view_history[i+1] is the view after
        # steps[i] was committed. Lets undo() just walk back one entry.
        self.view_history: list[str] = [base_view]
        self._registered = []  # keep refs to any registered pandas objects alive
 
    @classmethod
    def from_csv(cls, path: str, temp_directory: Optional[str] = None) -> "Pipeline":
        con = duckdb.connect(database=":memory:")
        if temp_directory:
            con.execute(f"SET temp_directory='{temp_directory}'")
        # read_csv is streamed/lazy under the hood; this does not load
        # the full file into Python.
        con.execute(f"CREATE VIEW raw_data AS SELECT * FROM read_csv_auto('{path}')")
        return cls(con, base_view="raw_data")
 
    @classmethod
    def from_parquet(cls, path: str, temp_directory: Optional[str] = None) -> "Pipeline":
        con = duckdb.connect(database=":memory:")
        if temp_directory:
            con.execute(f"SET temp_directory='{temp_directory}'")
        con.execute(f"CREATE VIEW raw_data AS SELECT * FROM read_parquet('{path}')")
        return cls(con, base_view="raw_data")
 
    @classmethod
    def from_excel(cls, path: str, sheet_name=0, temp_directory: Optional[str] = None) -> "Pipeline":
        """
        Excel files (.xlsx/.xls) are inherently capped at ~1M rows/sheet,
        so unlike CSV there's no out-of-core story here worth building —
        read via pandas/openpyxl and hand the dataframe to DuckDB.
        """
        import pandas as pd
 
        df = pd.read_excel(path, sheet_name=sheet_name)
        con = duckdb.connect(database=":memory:")
        if temp_directory:
            con.execute(f"SET temp_directory='{temp_directory}'")
        con.register("raw_data_df", df)
        con.execute("CREATE VIEW raw_data AS SELECT * FROM raw_data_df")
        pipeline = cls(con, base_view="raw_data")
        pipeline._registered.append(df)  # keep alive: duckdb only holds a reference
        return pipeline
 
    def columns(self) -> list[str]:
        return [c[0] for c in self.con.execute(f"DESCRIBE {self.current_view}").fetchall()]
 
    def profile(self) -> list[dict]:
        """
        Per-column summary: type, null %, distinct count, and (for numeric
        columns) mean/std/min/max — everything DuckDB's SUMMARIZE computes
        in one pass over the data, so this is safe on large files too.
        """
        rows = self.con.execute(f"SUMMARIZE {self.current_view}").fetchall()
        col_names = [d[0] for d in self.con.description]
        result = []
        for row in rows:
            entry = dict(zip(col_names, row))
            result.append({
                "column": entry.get("column_name"),
                "type": entry.get("column_type"),
                "min": entry.get("min"),
                "max": entry.get("max"),
                "mean": entry.get("avg"),
                "std": entry.get("std"),
                "distinct": entry.get("approx_unique"),
                "null_pct": entry.get("null_percentage"),
            })
        return result
 
    # -- EDA -----------------------------------------------------------
    # Numeric DuckDB types worth histogramming/correlating. Booleans and
    # dates are deliberately excluded — they're better shown as value
    # counts (bool) or left out of a correlation matrix (date) than
    # bucketed like a continuous variable.
    _NUMERIC_TYPES = {
        "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
        "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT",
        "FLOAT", "DOUBLE", "DECIMAL", "REAL",
    }
 
    def numeric_columns(self) -> list[str]:
        rows = self.con.execute(f"DESCRIBE {self.current_view}").fetchall()
        return [
            name for name, col_type, *_ in rows
            if col_type.split("(")[0].upper() in self._NUMERIC_TYPES
        ]
 
    def categorical_columns(self, max_distinct: int = 30) -> list[str]:
        """
        Non-numeric columns with a manageable number of distinct values —
        the ones a bar chart of value counts actually makes sense for.
        A free-text column with thousands of unique values is excluded;
        that's not what "categorical" means here.
        """
        numeric = set(self.numeric_columns())
        all_cols = self.columns()
        con = self.con
        current_view = self.current_view
        result = []
        for col in all_cols:
            if col in numeric:
                continue
            distinct = con.execute(
                f'SELECT approx_count_distinct("{col}") FROM {current_view}'
            ).fetchone()[0]
            if distinct <= max_distinct:
                result.append(col)
        return result
 
    def histogram(self, column: str, bins: int = 10) -> dict:
        """
        Equal-width histogram for a numeric column, computed in one
        bounded DuckDB query (no data pulled into Python row-by-row).
        Returns bin labels and counts, ready for a bar/histogram chart.
        """
        con = self.con
        current_view = self.current_view
 
        min_val, max_val = con.execute(
            f'SELECT MIN("{column}"), MAX("{column}") FROM {current_view} '
            f'WHERE "{column}" IS NOT NULL'
        ).fetchone()
 
        if min_val is None or max_val is None:
            return {"labels": [], "counts": []}
        if min_val == max_val:
            count = con.execute(
                f'SELECT COUNT(*) FROM {current_view} WHERE "{column}" = {min_val}'
            ).fetchone()[0]
            return {"labels": [f"{min_val:.2f}"], "counts": [count]}
 
        width = (max_val - min_val) / bins
        rows = con.execute(
            f'SELECT LEAST(CAST((("{column}" - {min_val}) / {width}) AS INTEGER), {bins - 1}) AS bucket, '
            f'COUNT(*) FROM {current_view} '
            f'WHERE "{column}" IS NOT NULL '
            f'GROUP BY bucket ORDER BY bucket'
        ).fetchall()
 
        counts_by_bucket = {b: c for b, c in rows}
        labels, counts = [], []
        for i in range(bins):
            lo = min_val + i * width
            hi = lo + width
            labels.append(f"{lo:.1f}–{hi:.1f}")
            counts.append(counts_by_bucket.get(i, 0))
        return {"labels": labels, "counts": counts}
 
    def value_counts(self, column: str, top_n: int = 15) -> dict:
        """Top-N most frequent values in a column, for a bar chart."""
        rows = self.con.execute(
            f'SELECT "{column}", COUNT(*) c FROM {self.current_view} '
            f'WHERE "{column}" IS NOT NULL '
            f'GROUP BY "{column}" ORDER BY c DESC LIMIT {top_n}'
        ).fetchall()
        return {"labels": [str(r[0]) for r in rows], "counts": [r[1] for r in rows]}
 
    def correlation_matrix(self, max_columns: int = 8) -> dict:
        """
        Pearson correlation between every pair of numeric columns, via
        DuckDB's corr() aggregate. Capped at max_columns to keep the
        pairwise query count (and the resulting chart) legible — with
        more numeric columns than that, a heatmap stops being readable
        anyway.
        """
        cols = self.numeric_columns()[:max_columns]
        truncated = len(self.numeric_columns()) > max_columns
        con = self.con
        current_view = self.current_view
 
        matrix = []
        for a in cols:
            row = []
            for b in cols:
                if a == b:
                    row.append(1.0)
                else:
                    val = con.execute(
                        f'SELECT corr("{a}", "{b}") FROM {current_view}'
                    ).fetchone()[0]
                    row.append(round(val, 3) if val is not None else None)
            matrix.append(row)
 
        return {"columns": cols, "matrix": matrix, "truncated": truncated}
 
    def plan(self, step: Step) -> StepResult:
        """Ask a step what it would do, without applying it."""
        result = step.plan(self)
        step._planned = result
        return result
 
    def preview(self, limit: int = 50):
        """Cheap look at the current state — never materializes the full table."""
        return self.con.execute(
            f"SELECT * FROM {self.current_view} LIMIT {limit}"
        ).fetchdf()
 
    def commit(self, step: Step) -> None:
        """
        Apply a previously-planned step: create a new view on top of the
        current one, and advance current_view to point at it.
        """
        result = step._planned or step.plan(self)
        step.applied_description = result.description
        self._view_counter += 1
        new_view = f"step_{self._view_counter}_{step.id}"
        self.con.execute(f"CREATE VIEW {new_view} AS {result.sql}")
        self.current_view = new_view
        self.steps.append(step)
        self.view_history.append(new_view)
 
    def can_undo(self) -> bool:
        return len(self.steps) > 0
 
    def undo(self) -> Optional[Step]:
        """
        Roll back the most recently committed step. Drops the view it
        created and moves current_view back to the previous one. Returns
        the undone Step (so a UI can say "undid: ...") or None if there
        was nothing to undo.
        """
        if not self.can_undo():
            return None
        undone_step = self.steps.pop()
        undone_view = self.view_history.pop()
        self.current_view = self.view_history[-1]
        self.con.execute(f"DROP VIEW IF EXISTS {undone_view}")
        return undone_step
 
    def row_count(self) -> int:
        return self.con.execute(f"SELECT COUNT(*) FROM {self.current_view}").fetchone()[0]
 
    def export_parquet(self, path: str) -> None:
        self.con.execute(f"COPY {self.current_view} TO '{path}' (FORMAT PARQUET)")
 
    def export_csv(self, path: str) -> None:
        self.con.execute(f"COPY {self.current_view} TO '{path}' (FORMAT CSV, HEADER)")
 
    def export_recipe(self) -> str:
        """Serialize the committed step chain as JSON — replayable on a new file."""
        return json.dumps([s.to_dict() for s in self.steps], indent=2)
 
 
if __name__ == "__main__":
    # Minimal smoke test — replace with a real file to sanity-check locally.
    print("Pipeline core module loaded OK.")
 