"""
Safe LangChain agent that exposes deterministic analysis functions as Tools.
No Python REPL execution, no allow_dangerous_code=True.

Usage:
    python langchain_agent_tools_safe_unified.py path/to/data.csv

Requirements:
    pip install pandas numpy matplotlib seaborn scipy tabulate scikit-learn duckdb
    pip install langchain langchain-experimental langchain-community langchain-ollama
    (optional) Ollama executable + ollama Python package to run Llama locally
"""

import sys
import os
import re
import json
import difflib
from pathlib import Path
import pandas as pd
import numpy as np

# Force non-interactive backend before importing pyplot to avoid Tcl/Tk errors on Windows
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tabulate import tabulate

# --------- Output folder ----------
OUTPUT_DIR = Path("analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Global default CSV path — set in main()
DEFAULT_CSV_PATH = None

# --------- Core analysis functions ----------
def load_data(path):
    return pd.read_csv(path)

def optimize_memory(df, verbose=True):
    start_mem = df.memory_usage(deep=True).sum() / 1024**2  # MB
    if verbose:
        print(f"Initial memory usage: {start_mem:.2f} MB")
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type == "float64":
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif col_type == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"Optimized memory usage: {end_mem:.2f} MB")
    return df

def descriptive_summary(df, max_columns_preview=10):
    out = {}
    out['shape'] = df.shape
    out['dtypes'] = df.dtypes.astype(str).to_dict()
    out['head'] = df.head().to_dict(orient='list')
    out['num_columns'] = df.select_dtypes(include='number').columns.tolist()
    out['cat_columns'] = df.select_dtypes(include='object').columns.tolist()
    try:
        out['describe'] = df.describe(include='all', datetime_is_numeric=True).T
    except TypeError:
        out['describe'] = df.describe(include='all').T
    out['missing_counts'] = df.isnull().sum().sort_values(ascending=False).head(max_columns_preview).to_dict()
    return out

def missing_value_report(df):
    miss = df.isnull().sum()
    pct = (miss / len(df) * 100).round(2)
    report = pd.DataFrame({'missing_count': miss, 'missing_pct': pct})
    report = report[report['missing_count'] > 0].sort_values('missing_pct', ascending=False)
    return report

def suggest_imputation(df):
    suggestions = {}
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            suggestions[col] = "mean/median or model-based imputation. If skewed, use median."
        else:
            nunique = df[col].nunique(dropna=True)
            if nunique < 10:
                suggestions[col] = "mode or 'missing' category. Consider imputation by group."
            else:
                suggestions[col] = "Consider using a model-based imputation or dropping if few missing."
    return suggestions

def duplicate_report(df):
    dup_count = df.duplicated().sum()
    duplicated_rows = df[df.duplicated(keep=False)]
    return dup_count, duplicated_rows

def outliers_iqr(df, column, k=1.5):
    series = df[column].dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    mask = (df[column] < lower) | (df[column] > upper)
    return mask, lower, upper, iqr

def outliers_zscore(df, column, thresh=3.0):
    series = df[column].dropna()
    z = (series - series.mean()) / series.std(ddof=0)
    mask = z.abs() > thresh
    return mask.reindex(df.index, fill_value=False)

def compute_correlations(df, method='pearson'):
    num = df.select_dtypes(include='number')
    if num.shape[1] < 2:
        return None
    corr = num.corr(method=method)
    return corr

def top_correlated_pairs(corr_df, top_k=10):
    # use upper triangle to avoid duplicate symmetric pairs
    corr = corr_df.copy()
    tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    stacked = tri.abs().stack().sort_values(ascending=False)
    top = stacked.head(top_k)
    return top

def save_heatmap(corr, filename="corr_heatmap.png"):
    import seaborn as sns
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
    path = OUTPUT_DIR / filename
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return str(path)

def plot_histogram_single(df, col):
    import seaborn as sns
    plt.figure(figsize=(6,3))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"Distribution: {col}")
    path = OUTPUT_DIR / f"hist_{col}.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return str(path)

def plot_scatter(df, x, y):
    import seaborn as sns
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x=x, y=y)
    p = OUTPUT_DIR / f"scatter_{x}_vs_{y}.png"
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    return str(p)

def generate_quick_report(df):
    lines = []
    lines.append(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    desc = descriptive_summary(df)
    lines.append("Numeric cols: " + ", ".join(desc['num_columns'][:10]))
    mv = missing_value_report(df)
    if not mv.empty:
        lines.append("Top missing: " + "; ".join([f"{idx}:{row['missing_count']}" for idx, row in mv.head(5).iterrows()]))
    pearson = compute_correlations(df, 'pearson')
    if pearson is not None:
        top = top_correlated_pairs(pearson, top_k=5)
        lines.append("Top correlations: " + "; ".join([f"{i[0]}-{i[1]}:{v:.2f}" for i, v in top.items()]))
    return "\n".join(lines)


# ----------------- Robust parsing helpers -----------------
def _strip_surrounding_quotes(s: str):
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s

def _normalize_dict_keys(d: dict):
    """Strip accidental quotes around keys like '"path"' -> 'path'"""
    new = {}
    for k, v in d.items():
        if isinstance(k, str):
            nk = _strip_surrounding_quotes(k)
        else:
            nk = k
        # also if value is a quoted string, strip outer quotes (keeps inner JSON intact)
        if isinstance(v, str):
            nv = _strip_surrounding_quotes(v)
        else:
            nv = v
        new[nk] = nv
    return new

def _try_parse_string_json(s: str):
    """Try to robustly parse a possibly double-encoded or quoted JSON string.
       Returns the parsed object (dict/list/str) or the original string if parse fails.
    """
    if not isinstance(s, str):
        return s
    s0 = s.strip()
    # try direct json.loads
    try:
        parsed = json.loads(s0)
        # If parsed is a string again, attempt one more decode (handles double encoding)
        if isinstance(parsed, str):
            try:
                parsed2 = json.loads(parsed)
                return parsed2
            except Exception:
                return parsed
        return parsed
    except Exception:
        # try stripping outer quotes then parse
        inner = _strip_surrounding_quotes(s0)
        if inner != s0:
            try:
                parsed = json.loads(inner)
                return parsed
            except Exception:
                return inner
        return s


# ----------------- Tool wrappers (file-based safe interface) -----------------
# ----------------- Robust parsing helpers (REPLACE existing helpers) -----------------
def _strip_surrounding_quotes(s: str):
    s = s.strip()
    # remove matching outer single or double quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s

def _normalize_tool_input(raw):
    """
    Normalize whatever the agent/runtime passes into a canonical Python object:
      - unwrap common wrapper keys (input, args, kwargs, tool_input)
      - handle list/tuple by taking first element
      - robustly parse strings (double-encoded JSON, single quotes)
      - normalize dict keys (strip accidental surrounding quotes)
    Returns: a Python dict/string suitable for _ensure_arg_path_flexible/_read_df_from_arg.
    """
    # unwrap lists/tuples
    if isinstance(raw, (list, tuple)) and len(raw) > 0:
        raw = raw[0]

    # unwrap common wrapper dict keys
    if isinstance(raw, dict):
        # common wrapper keys to check in order
        for key in ("input", "inputs", "args", "kwargs", "tool_input", "tool_kwargs", "data"):
            if key in raw and raw[key] is not None:
                raw = raw[key]
                break
        # if after unwrapping raw is still a dict with quoted keys, normalize
        if isinstance(raw, dict):
            raw = _normalize_dict_keys(raw)
            return raw

    # if it's a string, try robust parsing
    if isinstance(raw, str):
        parsed = _try_parse_string_json(raw)
        # if parsed is dict, normalize its keys
        if isinstance(parsed, dict):
            return _normalize_dict_keys(parsed)
        return parsed

    # otherwise return as-is
    return raw

def safe_tool_wrapper(func):
    """
    Return a callable that accepts *any* raw input from the agent,
    normalizes it, and calls the underlying tool function.
    The wrapper never raises — it returns a structured dict on error.
    """
    def inner(raw=None):
        try:
            normalized = _normalize_tool_input(raw)
            # ensure fallback DEFAULT_CSV_PATH is added if missing
            norm = _ensure_arg_path_flexible(normalized)
            # call the underlying tool function (which expects a normalized dict)
            result = func(norm)
            # If result is not dict, normalize to structured form
            if isinstance(result, dict):
                return result
            else:
                return {"message": str(result), "files": [], "meta": {}}
        except Exception as e:
            # return an informative but non-exploding response
            return {"message": f"Tool execution error: {e}", "files": [], "meta": {}}
    return inner


def _normalize_dict_keys(d: dict):
    """Return a new dict with keys cleaned from accidental surrounding quotes
       and with string values stripped of outer quotes too.
    """
    new = {}
    for k, v in d.items():
        nk = k
        if isinstance(k, str):
            nk = _strip_surrounding_quotes(k)
        nv = v
        if isinstance(v, str):
            nv = _strip_surrounding_quotes(v)
        new[nk] = nv
    return new

def _try_parse_string_json(s: str):
    """Robustly parse a string that might be:
       - a JSON object string using double quotes (valid JSON)
       - a JSON-like string using single quotes (e.g. \"{'path':'a.csv'}\")
       - double-encoded JSON (a JSON string whose value is another JSON string)
       - a plain path string
       Returns: parsed object (dict/list/str) or original string if not parseable.
    """
    if not isinstance(s, str):
        return s
    s0 = s.strip()

    # 1) Try standard JSON parse
    try:
        parsed = json.loads(s0)
        # If this yields a string (double-encoded), try load again
        if isinstance(parsed, str):
            try:
                parsed2 = json.loads(parsed)
                return parsed2
            except Exception:
                return parsed
        return parsed
    except Exception:
        pass

    # 2) Try replacing single quotes with double quotes (common non-JSON dict literal)
    try:
        s_double = s0.replace("'", '"')
        parsed = json.loads(s_double)
        return parsed
    except Exception:
        pass

    # 3) If outer string has extra quotes (like "\"{'path':'a.csv'}\"" or "'{\"path\":\"a.csv\"}'"),
    #    strip outer quotes and retry parsing inner content
    inner = _strip_surrounding_quotes(s0)
    if inner != s0:
        try:
            parsed = json.loads(inner)
            return parsed
        except Exception:
            # try single->double on inner
            try:
                inner2 = inner.replace("'", '"')
                parsed = json.loads(inner2)
                return parsed
            except Exception:
                return inner

    # 4) Give up: return original string
    return s0


# ----------------- Robust _read_df_from_arg (REPLACE existing implementation) -----------------
def _read_df_from_arg(arg):
    """
    Read dataframe from arg. Robust handling for:
      - None -> DEFAULT_CSV_PATH
      - plain path string
      - JSON string (including single-quoted dicts and double-encoded JSON)
      - dict with 'path' (robust to quoted keys like '\"path\"' or \"'path'\")
    """
    global DEFAULT_CSV_PATH

    path = None

    if arg is None:
        path = DEFAULT_CSV_PATH

    elif isinstance(arg, str):
        parsed = _try_parse_string_json(arg)
        # If parsed is a dict, normalize keys and read path
        if isinstance(parsed, dict):
            parsed = _normalize_dict_keys(parsed)
            path = parsed.get('path')
        else:
            # parsed might be a plain path string after cleaning
            if isinstance(parsed, str) and os.path.exists(parsed):
                path = parsed
            else:
                # fallback to original arg if it is a path
                path = arg if os.path.exists(arg) else None

    elif isinstance(arg, dict):
        clean = _normalize_dict_keys(arg)
        path = clean.get('path')

    # final fallback to default
    if path is None:
        path = DEFAULT_CSV_PATH

    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"CSV path not found: {path}")

    df = pd.read_csv(path)
    return df



# --- Helper: normalize different arg shapes ---
def _ensure_arg_path_flexible(maybe):
    global DEFAULT_CSV_PATH
    if maybe is None:
        return {"path": DEFAULT_CSV_PATH}
    if isinstance(maybe, dict):
        if 'path' not in maybe or not maybe.get('path'):
            maybe['path'] = DEFAULT_CSV_PATH
        return maybe
    if isinstance(maybe, (list, tuple)) and len(maybe) > 0:
        return _ensure_arg_path_flexible(maybe[0])
    if isinstance(maybe, str):
        parsed = _try_parse_string_json(maybe)
        if isinstance(parsed, dict):
            parsed = _normalize_dict_keys(parsed)
            if 'path' not in parsed or not parsed.get('path'):
                parsed['path'] = DEFAULT_CSV_PATH
            return parsed
        if os.path.exists(maybe):
            return {"path": maybe}
        return {"path": DEFAULT_CSV_PATH}
    return {"path": DEFAULT_CSV_PATH}

# --- Structured tool outputs: always return dict {'message', 'files'[], 'meta':{}} ---
def tool_describe(arg):
    df = _read_df_from_arg(arg)
    df = optimize_memory(df, verbose=False)
    desc = descriptive_summary(df)
    message = f"Shape: {desc['shape'][0]} rows x {desc['shape'][1]} cols\n"
    message += "Numeric cols: " + ", ".join(desc['num_columns'][:20]) + "\n"
    message += "Categorical cols: " + ", ".join(desc['cat_columns'][:20]) + "\n"

    head_path = OUTPUT_DIR / "head_preview.csv"
    df.head(20).to_csv(head_path, index=False)

    return {"message": message, "files": [str(head_path)], "meta": {"num_rows": desc['shape'][0], "num_cols": desc['shape'][1]}}

def tool_missing(arg):
    df = _read_df_from_arg(arg)
    df = optimize_memory(df, verbose=False)
    mv = missing_value_report(df)
    if mv.empty:
        return {"message": "No missing values detected.", "files": [], "meta": {}}
    path = OUTPUT_DIR / "missing_report.csv"
    mv.to_csv(path)
    return {"message": "Missing values report saved.", "files": [str(path)], "meta": {"num_missing_cols": mv.shape[0]}}

def tool_duplicates(arg):
    df = _read_df_from_arg(arg)
    dup_count, duplicated_rows = duplicate_report(df)
    msg = f"Duplicate rows count: {dup_count}"
    files = []
    if dup_count > 0:
        path = OUTPUT_DIR / "duplicated_rows_example.csv"
        duplicated_rows.head(100).to_csv(path, index=False)
        files.append(str(path))
        msg += f" — example saved to {path}"
    return {"message": msg, "files": files, "meta": {"dup_count": int(dup_count)}}

def tool_outliers(arg):
    if isinstance(arg, str):
        try:
            obj = _try_parse_string_json(arg)
        except Exception:
            raise ValueError("outliers tool expects JSON or dict with 'path' and 'col'")
    else:
        obj = arg
    df = _read_df_from_arg(obj)
    col = obj.get('col') if isinstance(obj, dict) else None
    if not col:
        return {"message": "Specify 'col' key with column name (e.g., {'path':'file.csv','col':'Age'})", "files": [], "meta": {}}
    if col not in df.columns:
        close = difflib.get_close_matches(col, df.columns.tolist(), n=5, cutoff=0.5)
        if close:
            return {"message": f"Column '{col}' not found. Did you mean: {', '.join(close)}?", "files": [], "meta": {"suggestions": close}}
        return {"message": f"Column '{col}' not found. Available columns: {', '.join(list(df.columns)[:20])}", "files": [], "meta": {}}
    mask, low, up, iqr = outliers_iqr(df, col)
    return {"message": f"Outliers (IQR rule) in {col}: {int(mask.sum())}, lower={low}, upper={up}", "files": [], "meta": {"outlier_count": int(mask.sum()), "lower": float(low), "upper": float(up)}}

def tool_correlations(arg):
    df = _read_df_from_arg(arg)
    df = optimize_memory(df, verbose=False)
    pearson = compute_correlations(df, 'pearson')
    if pearson is None:
        return {"message": "No numeric columns for correlation.", "files": [], "meta": {}}
    top = top_correlated_pairs(pearson, top_k=10)
    heatmap_path = save_heatmap(pearson, filename="pearson_heatmap.png")
    return {"message": "Top correlations:\n" + top.to_string(), "files": [heatmap_path], "meta": {}}

def tool_scatter(arg):
    if isinstance(arg, str):
        obj = _try_parse_string_json(arg)
    else:
        obj = arg
    df = _read_df_from_arg(obj)
    x = obj.get('x') if isinstance(obj, dict) else None
    y = obj.get('y') if isinstance(obj, dict) else None
    if not x or not y:
        return {"message": "Specify x and y: {'path':'file.csv','x':'col1','y':'col2'}", "files": [], "meta": {}}
    if x not in df.columns or y not in df.columns:
        return {"message": f"Columns not found. Available columns: {', '.join(list(df.columns)[:20])}", "files": [], "meta": {}}
    p = plot_scatter(df, x, y)
    return {"message": f"Scatter saved to: {p}", "files": [p], "meta": {}}

def tool_run_report(arg):
    df = _read_df_from_arg(arg)
    df = optimize_memory(df, verbose=False)
    report_text = generate_quick_report(df)
    path = OUTPUT_DIR / "quick_report_tool.txt"
    with open(path, "w", encoding="utf8") as f:
        f.write(report_text)
    return {"message": f"Quick report saved to: {path}\n{report_text}", "files": [str(path)], "meta": {}}


# ---------- Column-level describe tool (explicit, avoids ambiguous calls) ----------
def tool_describe_column(arg):
    """
    Expects arg = {'path': <path>, 'col': <column_name>}
    Returns structured dict.
    """
    if isinstance(arg, str):
        parsed = _try_parse_string_json(arg)
        if isinstance(parsed, dict):
            obj = parsed
        else:
            obj = {'path': arg}
    else:
        obj = arg
    obj = _normalize_dict_keys(obj) if isinstance(obj, dict) else obj
    df = _read_df_from_arg(obj)
    col = obj.get('col')
    if not col:
        return {"message": "Specify 'col' key, e.g. {'path': 'data.csv', 'col': 'Age'}", "files": [], "meta": {}}
    # fuzzy match
    if col not in df.columns:
        close = difflib.get_close_matches(col, df.columns.tolist(), n=5, cutoff=0.5)
        if close:
            return {"message": f"Column '{col}' not found. Did you mean: {', '.join(close)}?", "files": [], "meta": {"suggestions": close}}
        return {"message": f"Column '{col}' not found. Available columns: {', '.join(list(df.columns)[:20])}", "files": [], "meta": {}}

    out_lines = [f"Column: {col}", f"dtype: {df[col].dtype}"]
    if pd.api.types.is_numeric_dtype(df[col]):
        series = df[col].dropna()
        out_lines.append(f"count: {series.count()}, mean: {series.mean():.4f}, std: {series.std():.4f}, min: {series.min():.4f}, max: {series.max():.4f}")
        # save histogram
        hist_path = Path(plot_histogram_single(df, col))
        return {"message": "\n".join(out_lines), "files": [str(hist_path)], "meta": {"col": col}}
    else:
        out_lines.append(f"Top categories: {df[col].value_counts().head(10).to_dict()}")
        return {"message": "\n".join(out_lines), "files": [], "meta": {"col": col}}


# ---------- Flexible wrappers & agent creation ----------
def wrap_describe(*args, **kwargs):
    # accept many call styles; normalize what the agent actually passes
    if kwargs:
        raw = kwargs
    elif args:
        raw = args[0]
    else:
        raw = None
    print("DEBUG normalize_tool_input RAW:", repr(raw))
    normalized = _normalize_tool_input(raw)
    norm = _ensure_arg_path_flexible(normalized)
    return tool_describe(norm)

def wrap_missing(*args, **kwargs):
    if kwargs:
        arg = kwargs
    elif args:
        arg = args[0]
    else:
        arg = None
    norm = _ensure_arg_path_flexible(arg)
    return tool_missing(norm)

def wrap_duplicates(*args, **kwargs):
    if kwargs:
        arg = kwargs
    elif args:
        arg = args[0]
    else:
        arg = None
    norm = _ensure_arg_path_flexible(arg)
    return tool_duplicates(norm)

def wrap_outliers(*args, **kwargs):
    if kwargs:
        raw = kwargs
    elif args:
        raw = args[0]
    else:
        raw = None
    print("DEBUG normalize_tool_input RAW:", repr(raw))
    normalized = _normalize_tool_input(raw)
    norm = _ensure_arg_path_flexible(normalized)
    return tool_outliers(norm)

def wrap_correlations(*args, **kwargs):
    if kwargs:
        arg = kwargs
    elif args:
        arg = args[0]
    else:
        arg = None
    norm = _ensure_arg_path_flexible(arg)
    return tool_correlations(norm)

def wrap_scatter(*args, **kwargs):
    if kwargs:
        arg = kwargs
    elif args:
        arg = args[0]
    else:
        arg = None
    norm = _ensure_arg_path_flexible(arg)
    return tool_scatter(norm)

def wrap_run_report(*args, **kwargs):
    if kwargs:
        arg = kwargs
    elif args:
        arg = args[0]
    else:
        arg = None
    norm = _ensure_arg_path_flexible(arg)
    return tool_run_report(norm)

def wrap_describe_column(*args, **kwargs):
    if kwargs:
        raw = kwargs
    elif args:
        raw = args[0]
    else:
        raw = None
    print("DEBUG normalize_tool_input RAW:", repr(raw))
    normalized = _normalize_tool_input(raw)
    norm = _ensure_arg_path_flexible(normalized)
    return tool_describe_column(norm)

def create_agent(llm):
    try:
        from langchain.agents import initialize_agent, Tool, AgentType
    except Exception as e:
        raise RuntimeError("langchain agent imports failed. Install 'langchain' package.") from e

    tools = [
        Tool(
            name="describe",
            func=safe_tool_wrapper(tool_describe),
            description=(
                "Return dataset summary. Input optional. "
                "Examples: None, 'data.csv', '{\"path\":\"data.csv\"}'. "
                "To describe a single column, use the dedicated tool 'describe_column' with {'path':'data.csv','col':'Age'}."
            ),
        ),
        Tool(
            name="describe_column",
            func=safe_tool_wrapper(tool_describe_column),
            description=(
                "Describe a single column. Input: {'path':'data.csv','col':'Age'}. "
                "Returns a structured dict with 'message' and optional 'files' (e.g., histogram)."
            ),
        ),
        Tool(
            name="missing",
            func=wrap_missing,
            description=(
                "Return missing value report. Examples: None or '{\"path\":\"data.csv\"}'. "
                "Returns a dict with 'message' and 'files' (CSV of missing counts)."
            ),
        ),
        Tool(
            name="duplicates",
            func=wrap_duplicates,
            description=(
                "Detect duplicates. Input optional. Example: '{\"path\":\"data.csv\"}'."
            ),
        ),
        Tool(
            name="outliers",
            func=wrap_outliers,
            description=(
                "Detect outliers. Input must be JSON/dict with 'path' and 'col'. "
                "Example: '{\"path\":\"data.csv\",\"col\":\"Age\"}'"
            ),
        ),
        Tool(
            name="correlations",
            func=wrap_correlations,
            description=(
                "Compute correlations. Input optional. Returns heatmap file and top pairs."
            ),
        ),
        Tool(
            name="scatter",
            func=wrap_scatter,
            description=(
                "Create scatter plot. Accepts JSON/dict with 'path','x','y'. "
                "Example: '{\"path\":\"data.csv\",\"x\":\"col1\",\"y\":\"col2\"}'."
            ),
        ),
        Tool(
            name="run_report",
            func=wrap_run_report,
            description=(
                "Run quick automated report. Input optional. Returns report file and summary text."
            ),
        ),
    ]

    agent = initialize_agent(
    tools,
    llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    agent_kwargs={
        "max_iterations": 3,
        "return_intermediate_steps": False,
        "handle_parsing_errors": True,  # ✅ ensures the agent retries with your tool
        },
    )

    return agent


# ---------- Rule-based fallback executor (direct calls to functions) ----------
def match_column_by_name(name, columns, cutoff=0.5):
    # exact
    for c in columns:
        if c.lower() == name.lower():
            return c
    # substring
    for c in columns:
        if name.lower() in c.lower():
            return c
    # fuzzy
    close = difflib.get_close_matches(name, columns, n=1, cutoff=cutoff)
    return close[0] if close else None

def rule_based_executor(query, csv_path, df):
    q = query.strip()
    ql = q.lower()

    # Describe column
    # Describe column (robust matching: exact, substring, then fuzzy)
    if re.search(r"\b(describe|summary|show me the first|head|preview)\b", ql):
        # 1) Try exact token match for full column name (case-insensitive)
        for col in df.columns:
            if re.search(r"\b" + re.escape(col.lower()) + r"\b", ql):
                return tool_describe_column(json.dumps({"path": csv_path, "col": col}))

        # 2) Try substring match (e.g., "weight" matches "Weight (kg)")
        for col in df.columns:
            if col.lower() in ql or any(tok in col.lower() for tok in re.findall(r"[A-Za-z0-9_]+", ql)):
                # If the query contains a short token that is inside the column name, accept it
                # prefer the first good match
                return tool_describe_column(json.dumps({"path": csv_path, "col": col}))

        # 3) Fuzzy match (close names) using difflib
        user_tokens = re.findall(r"[A-Za-z0-9_()%-]+", q)
        # flatten tokens to one string to compare against column names
        query_joined = " ".join(user_tokens).lower()
        close = difflib.get_close_matches(query_joined, df.columns.tolist(), n=1, cutoff=0.5)
        if not close:
            # try token-wise fuzzy match
            for tok in user_tokens:
                c2 = difflib.get_close_matches(tok.lower(), df.columns.tolist(), n=1, cutoff=0.6)
                if c2:
                    close = c2
                    break
        if close:
            return tool_describe_column(json.dumps({"path": csv_path, "col": close[0]}))

        # fallback: dataset summary
        return tool_describe(csv_path)

    
    # if re.search(r"\b(describe|summary|show me the first|head|preview)\b", ql):
    #     # try match a column mentioned in the query
    #     for col in df.columns:
    #         if re.search(r"\b" + re.escape(col.lower()) + r"\b", ql):
    #             # column-level summary via new tool
    #             return tool_describe_column(json.dumps({"path": csv_path, "col": col}))
    #     # fallback: dataset summary
    #     return tool_describe(csv_path)

    # Missing values
    if re.search(r"\b(missing|null|na|missing values)\b", ql):
        return tool_missing(csv_path)

    # Duplicates
    if re.search(r"\b(duplicate|duplicates)\b", ql):
        return tool_duplicates(csv_path)

    # Outliers: "outliers in Age" or "find outliers Age"
    m = re.search(r"(outlier|outliers).*(for|in|of)?\s*(?P<col>[A-Za-z0-9_ ()%-]+)", q, flags=re.I)
    if m:
        col = m.group("col").strip()
        col_matched = None
        for c in df.columns:
            if c.lower() == col.lower() or col.lower() in c.lower():
                col_matched = c
                break
        if col_matched:
            return tool_outliers(json.dumps({"path": csv_path, "col": col_matched}))
        else:
            close = difflib.get_close_matches(col, df.columns.tolist(), n=5, cutoff=0.5)
            if close:
                return {"message": f"Could not find exact column '{col}'. Did you mean: {', '.join(close)}?", "files": [], "meta": {"suggestions": close}}
            return {"message": f"Could not find a column named '{col}' in dataset.", "files": [], "meta": {"available_columns": list(df.columns[:20])}}

    # Correlation between two columns
    # Correlation between or for two columns (robust phrasing)
    m = re.search(
        r"correlation\s*(?:between|for|of)?\s+(?P<c1>[A-Za-z0-9_ ()%-]+)\s+(and|,)\s+(?P<c2>[A-Za-z0-9_ ()%-]+)",
        q,
        flags=re.I,
    )
    if m:
        c1 = m.group("c1").strip()
        c2 = m.group("c2").strip()

        def match_col(name):
            for c in df.columns:
                if c.lower() == name.lower() or name.lower() in c.lower():
                    return c
            return None

        mc1 = match_col(c1)
        mc2 = match_col(c2)

        if mc1 and mc2:
            corr_val = df[[mc1, mc2]].corr().iloc[0, 1]
            return {
                "message": f"Correlation between {mc1} and {mc2}: {corr_val:.4f}",
                "files": [],
                "meta": {"cols": [mc1, mc2], "corr": corr_val},
            }
        else:
            return {
                "message": f"Could not match both columns ({c1}, {c2}).",
                "files": [],
                "meta": {},
            }


    # Scatter: find two columns mentioned
    if re.search(r"\b(scatter|plot)\b", ql):
        found = []
        for c in df.columns:
            if c.lower() in ql:
                found.append(c)
        if len(found) >= 2:
            x, y = found[0], found[1]
            p = plot_scatter(df, x, y)
            return {"message": f"Saved scatter plot: {p}", "files": [p], "meta": {"x": x, "y": y}}
        else:
            return {"message": "Please mention two column names to scatter, e.g. 'scatter Age vs Calories'.", "files": [], "meta": {}}

    # Run quick report
    if re.search(r"\b(run report|quick report|full report|run the report)\b", ql):
        return tool_run_report(csv_path)

    return {"message": ("I didn't understand exactly which analysis you want.\n"
            "Try one of: describe <column?>, missing, duplicates, outliers <column>, "
            "correlation between <col1> and <col2>, scatter <col1> <col2>, run_report"), "files": [], "meta": {}}


# -------------- Main ----------------
def main(csv_path):
    global DEFAULT_CSV_PATH
    DEFAULT_CSV_PATH = csv_path  # set global fallback

    print("CSV:", csv_path)
    print("Loading (and optimizing memory)...")
    df = load_data(csv_path)
    df = optimize_memory(df, verbose=True)
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # Try to initialize an LLM (prefer local Ollama from langchain-community)
    llm = None
    try:
        try:
            from langchain_community.llms import Ollama
        except Exception:
            from langchain_ollama import OllamaLLM as Ollama
        llm = Ollama(model="llama3")
        print("Using local Ollama Llama3 model (if Ollama service + model present).")
    except Exception as e:
        print("Local Ollama not available or failed to initialize:", e)
        llm = None

    # create agent if llm available, otherwise run fallback CLI using tools
        # Instead of creating a LangChain agent (tool-parsing), use a deterministic loop:
    if llm is not None:
        print("\nLLM available but running in safe mode (agent tool parsing disabled).")
        print("I'll run deterministic rule-based tools first; for other questions I'll ask the LLM for text replies.")
        print("Type 'exit' to quit.")
        while True:
            q = input(">>> ").strip()
            if q.lower() in ("exit", "quit", "q"):
                break
            if not q:
                continue

            # 1) Try deterministic rule-based executor first (no LangChain parsing)
            try:
                resp = rule_based_executor(q, csv_path, df)
                # rule_based_executor returns a dict structured response
                if isinstance(resp, dict):
                    print("\n", resp.get('message'), "\n")
                    if resp.get('files'):
                        print("Files:", ", ".join(resp.get('files')))
                else:
                    # fallback prints a plain string
                    print("\n", resp, "\n")
                # If rule_based_executor provided a concrete result (not the generic fallback), continue
                # We'll treat the generic fallback message as 'not handled' and send to LLM
                if isinstance(resp, dict) and resp.get('meta'):
                    # some meta indicates a real result -> continue to next prompt
                    continue
                # If the response message is the generic "I didn't understand..." then fallthrough to LLM
                if isinstance(resp, dict) and resp.get('message') and "I didn't understand" not in resp.get('message'):
                    continue
            except Exception as e:
                # if something went wrong in deterministic path, continue to LLM
                print("Deterministic executor error (falling back to LLM):", e)

            # 2) If not handled deterministically, ask the LLM for a plain-text reply (NO tools)
            try:
                # Use the LLM directly — the exact wrapper call depends on your LLM class
                # This tries a few common method names to be compatible with different wrappers:
                llm_result = None
                if hasattr(llm, "generate"):
                    # newest LangChain LLM API: generate
                    # construct a simple prompt
                    prompt = f"You are an assistant. The user asked: {q}\nIf this is a dataset analysis request (like 'describe Age' or 'outliers Age'), reply: 'Please use the exact command: describe_column with {{\"col\":\"Age\"}}' or similar. Otherwise answer succinctly."
                    out = llm.generate([{"role": "user", "content": prompt}])
                    # `out.generations` structure varies by LLM — attempt to extract text safely
                    try:
                        llm_result = out.generations[0][0].text
                    except Exception:
                        # fallback: stringify
                        llm_result = str(out)
                elif hasattr(llm, "predict"):
                    prompt = f"You are an assistant. The user asked: {q}\nAnswer succinctly. If it's a dataset request use one-line instruction to run a tool."
                    llm_result = llm.predict(prompt)
                elif hasattr(llm, "call"):
                    prompt = f"You are an assistant. The user asked: {q}\nAnswer succinctly."
                    r = llm.call({"input": prompt})
                    # try common keys
                    llm_result = r.get("text") if isinstance(r, dict) else str(r)
                else:
                    # fallback: simply echo
                    llm_result = f"(LLM available but unknown call interface) User asked: {q}"
                print("\nLLM:", llm_result, "\n")
            except Exception as e:
                print("LLM call failed:", e)

    else:
        print("No local LLM found — using fallback CLI mode (no LLM).")
        fallback_cli(csv_path)
        
    


# Fallback CLI to call tools directly when LLM not available
def fallback_cli(csv_path):
    print("Fallback CLI mode. Type commands:")
    print(" describe | describe_column <col> | missing | duplicates | outliers <col> | corr | scatter <x> <y> | run_report | exit")
    while True:
        cmd = input(">>> ").strip().split()
        if not cmd:
            continue
        c = cmd[0].lower()
        if c in ("exit", "quit"):
            break
        try:
            if c == "describe":
                resp = tool_describe(csv_path)
                print(resp.get('message'))
                if resp.get('files'):
                    print("Files:", ", ".join(resp.get('files')))
            elif c == "describe_column":
                if len(cmd) < 2:
                    print("usage: describe_column <col>")
                else:
                    resp = tool_describe_column(json.dumps({'path': csv_path, 'col': cmd[1]}))
                    print(resp.get('message'))
                    if resp.get('files'):
                        print("Files:", ", ".join(resp.get('files')))
            elif c == "missing":
                resp = tool_missing(csv_path)
                print(resp.get('message'))
                if resp.get('files'):
                    print("Files:", ", ".join(resp.get('files')))
            elif c == "duplicates":
                resp = tool_duplicates(csv_path)
                print(resp.get('message'))
                if resp.get('files'):
                    print("Files:", ", ".join(resp.get('files')))
            elif c == "outliers":
                if len(cmd) < 2:
                    print("usage: outliers <col>")
                else:
                    resp = tool_outliers(json.dumps({'path': csv_path, 'col': cmd[1]}))
                    print(resp.get('message'))
            elif c == "corr":
                resp = tool_correlations(csv_path)
                print(resp.get('message'))
                if resp.get('files'):
                    print("Files:", ", ".join(resp.get('files')))
            elif c == "scatter":
                if len(cmd) < 3:
                    print("usage: scatter <x> <y>")
                else:
                    resp = tool_scatter(json.dumps({'path': csv_path, 'x': cmd[1], 'y': cmd[2]}))
                    print(resp.get('message'))
                    if resp.get('files'):
                        print("Files:", ", ".join(resp.get('files')))
            elif c == "run_report":
                resp = tool_run_report(csv_path)
                print(resp.get('message'))
                if resp.get('files'):
                    print("Files:", ", ".join(resp.get('files')))
            else:
                print("Unknown command:", c)
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python langchain_agent_tools_safe_unified.py path/to/data.csv")
        sys.exit(1)
    
    main(sys.argv[1])
    
