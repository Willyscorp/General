# data_analysis_report.py


import sys, os, math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate

OUTPUT_DIR = Path("analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data(path):
    df = pd.read_csv(path)
    return df

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
        print(f"Memory reduced by {(start_mem - end_mem) / start_mem * 100:.1f}%")

    return df


# ---------- DESCRIPTIVE ----------
def descriptive_summary(df, max_columns_preview=10):
    out = {}
    out['shape'] = df.shape
    out['dtypes'] = df.dtypes.astype(str).to_dict()
    out['head'] = df.head().to_dict(orient='list')
    out['num_columns'] = df.select_dtypes(include='number').columns.tolist()
    out['cat_columns'] = df.select_dtypes(include='object').columns.tolist()

    # call describe() in a safe way (backwards compatible)
    try:
        # newer pandas supports datetime_is_numeric
        out['describe'] = df.describe(include='all', datetime_is_numeric=True).T
    except TypeError:
        # older pandas does not accept datetime_is_numeric
        out['describe'] = df.describe(include='all').T

    # Basic distributions for numeric columns
    num = df.select_dtypes(include='number')
    out['missing_counts'] = df.isnull().sum().sort_values(ascending=False).head(max_columns_preview).to_dict()
    return out


# ---------- MISSING VALUES ----------
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

# ---------- DUPLICATES ----------
def duplicate_report(df):
    dup_count = df.duplicated().sum()
    duplicated_rows = df[df.duplicated(keep=False)]
    return dup_count, duplicated_rows

# ---------- OUTLIERS ----------
def outliers_iqr(df, column, k=1.5):
    """Return boolean mask of outliers by IQR rule."""
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

# ---------- CORRELATIONS ----------
def compute_correlations(df, method='pearson'):
    num = df.select_dtypes(include='number')
    if num.shape[1] < 2:
        return None
    corr = num.corr(method=method)
    return corr

def top_correlated_pairs(corr_df, top_k=10):
    # flatten correlations, exclude self pairs
    corr = corr_df.copy()
    np.fill_diagonal(corr.values, np.nan)
    stacked = corr.abs().stack().sort_values(ascending=False)
    top = stacked.head(top_k)
    return top

# ---------- HYPOTHESIS TESTING ----------
def t_test_between_groups(df, numeric_col, group_col, groupA, groupB, equal_var=False):
    from scipy import stats  # lazy import
    a = df[df[group_col] == groupA][numeric_col].dropna()
    b = df[df[group_col] == groupB][numeric_col].dropna()
    tstat, pval = stats.ttest_ind(a, b, equal_var=equal_var, nan_policy='omit')
    return tstat, pval, len(a), len(b)

def mannwhitney_between_groups(df, numeric_col, group_col, groupA, groupB):
    from scipy import stats  # lazy import
    a = df[df[group_col] == groupA][numeric_col].dropna()
    b = df[df[group_col] == groupB][numeric_col].dropna()
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    return u, p, len(a), len(b)

def anova_oneway(df, numeric_col, group_col):
    from scipy import stats  # lazy import
    groups = [g[numeric_col].dropna() for _, g in df.groupby(group_col)]
    fstat, pval = stats.f_oneway(*groups)
    return fstat, pval, [len(g) for g in groups]

def chi_square_categorical(df, col1, col2):
    from scipy import stats  # lazy import
    ct = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    return chi2, p, dof, expected


# ---------- PLOTTING ----------
def save_heatmap(corr, filename="corr_heatmap.png"):
    import seaborn as sns  # lazy import
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
    path = OUTPUT_DIR / filename
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

def plot_histograms(df, numeric_cols, max_plots=10):
    import seaborn as sns  # lazy import
    cols = numeric_cols[:max_plots]
    for c in cols:
        plt.figure(figsize=(6,3))
        sns.histplot(df[c].dropna(), kde=True)
        plt.title(f"Distribution: {c}")
        p = OUTPUT_DIR / f"hist_{c}.png"
        plt.tight_layout()
        plt.savefig(p)
        plt.close()

def plot_scatter(df, x, y):
    import seaborn as sns  # lazy import
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x=x, y=y)
    p = OUTPUT_DIR / f"scatter_{x}_vs_{y}.png"
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    return p


# ---------- REPORT WRITING ----------
def save_text_report(text, filename="report.txt"):
    p = OUTPUT_DIR / filename
    with open(p, "w", encoding="utf8") as f:
        f.write(text)
    return p

# ---------- MAIN WORKFLOW ----------
def run_analysis(path):
    df = load_data(path)

    # --- Optimize memory usage before analysis ---
    df = optimize_memory(df, verbose=True)

    n_rows, n_cols = df.shape
    report_lines = []
    report_lines.append(f"Data loaded (after optimization): {n_rows} rows, {n_cols} columns\n")

    # descriptive
    desc = descriptive_summary(df)
    report_lines.append("Column types:\n" + tabulate([(k, v) for k,v in desc['dtypes'].items()], headers=["column","dtype"]))
    report_lines.append("\nTop missing columns:\n" + str(desc['missing_counts']))

    # missing values
    miss = missing_value_report(df)
    if not miss.empty:
        report_lines.append("\nMissing values (column, count, pct):\n" + miss.to_string())
        suggestions = suggest_imputation(df)
        report_lines.append("\nImputation suggestions:\n" + "\n".join([f"{k}: {v}" for k,v in suggestions.items()]))
    else:
        report_lines.append("\nNo missing values detected.")

    # duplicates
    dup_count, duplicated_rows = duplicate_report(df)
    report_lines.append(f"\nDuplicate rows count: {dup_count}")
    if dup_count > 0:
        report_lines.append(f"Example duplicated rows saved to file.")
        duplicated_rows.head(10).to_csv(OUTPUT_DIR / "duplicated_rows_example.csv", index=False)

    # numeric analysis
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if num_cols:
        report_lines.append("\nNumeric columns: " + ", ".join(num_cols))
        # outliers
        outlier_summary = []
        for c in num_cols:
            mask, low, up, iqr = outliers_iqr(df, c)
            n_out = mask.sum()
            outlier_summary.append((c, int(n_out), float(low), float(up)))
        report_lines.append("\nOutlier summary (IQR rule):\n" + tabulate(outlier_summary, headers=["col","n_outliers","lower","upper"]))
        # z-score outliers (count)
        z_summary = []
        for c in num_cols:
            mask_z = outliers_zscore(df, c)
            z_summary.append((c, int(mask_z.sum())))
        report_lines.append("\nOutlier summary (z-score>3):\n" + tabulate(z_summary, headers=["col","n_outliers_z"]))

        # correlations
        pearson = compute_correlations(df, method='pearson')
        if pearson is not None:
            top_pairs = top_correlated_pairs(pearson, top_k=15)
            report_lines.append("\nTop absolute Pearson correlations:\n" + top_pairs.to_string())
            hm = save_heatmap(pearson, filename="pearson_heatmap.png")
            report_lines.append(f"Pearson heatmap saved to: {hm}")

            # also Spearman
            spearman = compute_correlations(df, method='spearman')
            sp_top = top_correlated_pairs(spearman, top_k=15)
            report_lines.append("\nTop absolute Spearman correlations:\n" + sp_top.to_string())
            save_heatmap(spearman, filename="spearman_heatmap.png")
    else:
        report_lines.append("\nNo numeric columns for correlation/outlier analysis.")

    # quick hypothesis tests examples (auto-detect a candidate target & a categorical grouping)
    # pick first numeric column as target, and first categorical col as group (if available)
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    if num_cols and cat_cols:
        target = num_cols[0]
        group = cat_cols[0]
        report_lines.append(f"\nAutomatically running example tests with target='{target}' and group='{group}'")
        # if only two groups -> t-test
        groups = df[group].dropna().unique()
        if len(groups) == 2:
            a, b = groups[:2]
            tstat, pval, na, nb = t_test_between_groups(df, target, group, a, b)
            mw_u, mw_p, _, _ = mannwhitney_between_groups(df, target, group, a, b)
            report_lines.append(f"T-test between '{a}' and '{b}' on '{target}': t={tstat:.4f}, p={pval:.4f}, n={na},{nb}")
            report_lines.append(f"Mann-Whitney U: u={mw_u:.4f}, p={mw_p:.4f}")
        elif len(groups) > 2:
            fstat, pval, counts = anova_oneway(df, target, group)
            report_lines.append(f"ANOVA across groups on '{target}': F={fstat:.4f}, p={pval:.4f}, counts={counts}")
    else:
        report_lines.append("\nNot enough numeric+categorical columns to auto-run group tests.")

    # categorical association example (pick two categorical)
    if len(cat_cols) >= 2:
        c1, c2 = cat_cols[:2]
        try:
            chi2, p, dof, expected = chi_square_categorical(df, c1, c2)
            report_lines.append(f"\nChi-square test between '{c1}' and '{c2}': chi2={chi2:.4f}, p={p:.4f}, dof={dof}")
        except Exception as e:
            report_lines.append(f"\nChi-square failed between {c1},{c2}: {e}")

    # plots
    if num_cols:
        plot_histograms(df, num_cols, max_plots=6)
        # scatter the top correlated pair if exists
        if pearson is not None:
            top = top_correlated_pairs(pearson, top_k=1)
            if not top.empty:
                # top index is a multiindex (col1,col2)
                pair = top.index[0]
                x,y = pair
                p = plot_scatter(df, x, y)
                report_lines.append(f"Top correlated scatter saved to: {p}")

    # save report
    report_text = "\n\n".join(report_lines)
    rpt = save_text_report(report_text, filename="quick_report.txt")
    print("Analysis saved to:", rpt)
    print("Outputs in folder:", OUTPUT_DIR.resolve())
    return rpt

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_analysis_report.py data.csv")
        sys.exit(1)
    csv_path = sys.argv[1]
    run_analysis(csv_path)
