# Student Placement Analysis and Prediction
# Project 14 - Statistical and Mathematical Techniques
#
# UniPlace AI - University Placement Intelligence System
#   Original terminal run: python stu_place_final.py

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", context="notebook")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from scipy.stats import ttest_ind, pearsonr, spearmanr

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "student_placement.csv"

numerical_cols = [
    "CGPA", "Attendance", "Study_Hours", "Programming_Score",
    "Communication_Score", "Aptitude_Score", "Projects",
    "Backlogs", "Salary"
]
categorical_cols = ["Gender", "Internship", "Placement_Status"]

features = [
    "CGPA", "Attendance", "Study_Hours", "Programming_Score",
    "Communication_Score", "Aptitude_Score", "Projects", "Backlogs"
]
placement_features = [
    "CGPA", "Attendance", "Study_Hours", "Programming_Score",
    "Communication_Score", "Aptitude_Score", "Internship",
    "Projects", "Backlogs"
]

# ============================================================
# SHARED LOGIC (used by both the terminal run and the web app)
# ============================================================

def load_data(path=DATA_FILE):
    return pd.read_csv(path)


def clean_data(df):
    """TASK 1: remove duplicates, median/mode imputation."""
    df = df.drop_duplicates().copy()
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def probability_stats(df):
    """TASK 3: probability + Bayes theorem."""
    p_placed = (df["Placement_Status"] == "Placed").mean()
    p_cgpa_8 = (df["CGPA"] > 8).mean()
    high_cgpa = df[df["CGPA"] > 8]
    p_placed_given_cgpa = (high_cgpa["Placement_Status"] == "Placed").mean()
    p_high_cgpa_given_placed = (
        df[df["Placement_Status"] == "Placed"]["CGPA"].gt(8).mean()
    )
    bayes = (p_high_cgpa_given_placed * p_placed) / p_cgpa_8
    return {
        "p_placed": p_placed, "p_cgpa_8": p_cgpa_8,
        "p_placed_given_cgpa": p_placed_given_cgpa, "bayes": bayes,
    }


def internship_ttest(df):
    """TASK 4: salary of placed students, internship vs none (Welch t-test)."""
    placed = df[df["Placement_Status"] == "Placed"]
    with_i = placed[placed["Internship"] == "Yes"]["Salary"]
    without_i = placed[placed["Internship"] == "No"]["Salary"]
    t_stat, p_value = ttest_ind(with_i, without_i, equal_var=False)
    return t_stat, p_value, with_i.mean(), without_i.mean()


def correlations(df):
    """TASK 5: Pearson and Spearman matrices."""
    return (df[numerical_cols].corr(method="pearson"),
            df[numerical_cols].corr(method="spearman"))


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def _fit_salary_model(df, cols):
    reg_data = df[df["Placement_Status"] == "Placed"]
    X_train, X_test, y_train, y_test = train_test_split(
        reg_data[cols], reg_data["Salary"], test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {"model": model, "X_test": X_test, "y_test": y_test,
            "y_pred": y_pred, "metrics": regression_metrics(y_test, y_pred)}


def fit_simple_regression(df):
    """TASK 6: CGPA -> Salary."""
    return _fit_salary_model(df, ["CGPA"])


def fit_multiple_regression(df):
    """TASK 7: all features -> Salary."""
    return _fit_salary_model(df, features)


def gradient_descent(df, learning_rate=0.01, epochs=1000):
    """TASK 9: gradient descent on standardized CGPA -> Salary."""
    reg_data = df[df["Placement_Status"] == "Placed"]
    x = reg_data["CGPA"].values
    y = reg_data["Salary"].values
    x = (x - x.mean()) / x.std()
    m, b, n = 0, 0, len(x)
    cost_history = []
    for i in range(epochs):
        error = (m * x + b) - y
        cost_history.append(np.mean(error ** 2))
        m = m - learning_rate * (2 / n) * np.sum(error * x)
        b = b - learning_rate * (2 / n) * np.sum(error)
    return m, b, cost_history


def _encode_placement_X(data):
    X = data[placement_features].copy()
    X["Internship"] = X["Internship"].map({"Yes": 1, "No": 0})
    return X


def fit_placement_model(df):
    """TASK 10: logistic regression placement model."""
    X_place = _encode_placement_X(df)
    y_place = df["Placement_Status"].map({"Placed": 1, "Not Placed": 0})
    X_train, X_test, y_train, y_test = train_test_split(
        X_place, y_place, test_size=0.2, random_state=42, stratify=y_place
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    return {
        "model": model, "scaler": scaler,
        "metrics": {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        },
        "confusion": confusion_matrix(y_test, y_pred),
        "test_not_placed": int((y_test == 0).sum()),
    }


def placement_probability(bundle, data):
    """P(placed) for rows that contain the placement features."""
    X = bundle["scaler"].transform(_encode_placement_X(data))
    return bundle["model"].predict_proba(X)[:, 1]


# Support / priority indicator (built on the placement model above)
HIGH_CUT, MED_CUT = 97.5, 99.0   # probability % cut-offs (dataset-specific)


def tier_series(prob_pct, not_placed=None):
    prob_pct = np.asarray(prob_pct, dtype=float)
    high = prob_pct < HIGH_CUT
    if not_placed is not None:
        high = high | np.asarray(not_placed)
    return np.select([high, prob_pct < MED_CUT], ["High", "Medium"], "Low")


def focus_flags(d):
    return pd.DataFrame({
        "CGPA below 7": d["CGPA"] < 7,
        "2+ backlogs": d["Backlogs"] >= 2,
        "Attendance below 75%": d["Attendance"] < 75,
        "Communication below 70": d["Communication_Score"] < 70,
        "Aptitude below 70": d["Aptitude_Score"] < 70,
        "Programming below 80": d["Programming_Score"] < 80,
        "Fewer than 2 projects": d["Projects"] < 2,
        "No internship": d["Internship"] == "No",
    })


def support_table(df, bundle):
    out = df.copy()
    prob = placement_probability(bundle, df) * 100
    out["Placement_Probability"] = prob.round(1)
    out["Support_Priority"] = tier_series(
        prob, (df["Placement_Status"] == "Not Placed").to_numpy())
    flags = focus_flags(df)
    out["Focus_Areas"] = flags.apply(
        lambda r: ", ".join(flags.columns[r.to_numpy()]) or "-", axis=1)
    return out


# ============================================================
# ORIGINAL TERMINAL ANALYSIS (python stu_place_final.py)
# ============================================================

def _finish_plot(title, xlabel, ylabel, legend=False):
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=18, fontweight="bold")
    if legend:
        plt.legend()
    plt.tight_layout()
    plt.show()


def run_terminal():
    # TASK 1: DATA PREPROCESSING
    raw = load_data()
    print("\n--- ORIGINAL DATA ---")
    print(raw.head())
    print("\nShape:", raw.shape)
    print("\nMissing values:")
    print(raw.isnull().sum())
    print("\nDuplicate records:", raw.duplicated().sum())

    df = clean_data(raw)
    print("\nAfter cleaning:")
    print("Shape:", df.shape)
    print("Missing values:")
    print(df.isnull().sum())
    df.to_csv(BASE_DIR / "student_placement_cleaned.csv", index=False)

    # TASK 2: DESCRIPTIVE STATISTICS
    print("\n\n--- DESCRIPTIVE STATISTICS ---")
    print("\nMean:\n", df[numerical_cols].mean())
    print("\nMedian:\n", df[numerical_cols].median())
    print("\nMode:\n", df[numerical_cols].mode().iloc[0])
    print("\nVariance:\n", df[numerical_cols].var())
    print("\nStandard deviation:\n", df[numerical_cols].std())
    print("\nMinimum:\n", df[numerical_cols].min())
    print("\nMaximum:\n", df[numerical_cols].max())

    plt.figure(figsize=(10, 6))
    sns.histplot(df["CGPA"], bins=15, kde=True, color="#6C5CE7",
                 edgecolor="white", alpha=0.85)
    _finish_plot("CGPA Distribution of Students", "CGPA", "Number of Students")

    # TASK 3: PROBABILITY
    print("\n--- PROBABILITY ---")
    ps = probability_stats(df)
    print("P(Placement) =", round(ps["p_placed"], 4))
    print("P(CGPA > 8) =", round(ps["p_cgpa_8"], 4))
    print("P(Placement | CGPA > 8) =", round(ps["p_placed_given_cgpa"], 4))
    print("Bayes P(Placement | CGPA > 8) =", round(ps["bayes"], 4))

    print("\nLaw of Large Numbers:")
    for n in [10, 50, 100, 250, 500, len(df)]:
        sample = df.sample(n=n, random_state=42)
        probability = (sample["Placement_Status"] == "Placed").mean()
        print("Sample size =", n,
              "Placement probability =", round(probability, 4))

    # TASK 4: HYPOTHESIS TESTING
    print("\n--- HYPOTHESIS TESTING ---")
    t_stat, p_value, _, _ = internship_ttest(df)
    print("T-statistic =", round(t_stat, 4))
    print("P-value =", round(p_value, 4))
    print("Result: Significant difference" if p_value < 0.05
          else "Result: No significant difference")

    # TASK 5: CORRELATION
    print("\n--- CORRELATION ANALYSIS ---")
    pearson_corr, spearman_corr = correlations(df)
    print("\nPearson Correlation Matrix:")
    print(pearson_corr.round(3))
    print("\nSpearman Correlation Matrix:")
    print(spearman_corr.round(3))
    print("\nCorrelation with Salary:")
    print(pearson_corr["Salary"].sort_values(ascending=False).round(3))

    plt.figure(figsize=(12, 9))
    sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap="RdYlBu_r",
                center=0, linewidths=0.8, square=True,
                cbar_kws={"label": "Correlation"})
    plt.title("Pearson Correlation Matrix", fontsize=18, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # TASK 6: SIMPLE LINEAR REGRESSION
    print("\n--- SIMPLE LINEAR REGRESSION ---")
    simple = fit_simple_regression(df)
    sm, sx, sy = simple["model"], simple["X_test"], simple["y_test"]
    print("Slope =", round(sm.coef_[0], 4))
    print("Intercept =", round(sm.intercept_, 4))
    for k, v in simple["metrics"].items():
        print(k, "=", round(v, 4))

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=sx["CGPA"], y=sy, s=75, alpha=0.75, color="#0984E3",
                    edgecolor="white", linewidth=0.6, label="Actual Students")
    order = np.argsort(sx["CGPA"].values)
    plt.plot(sx["CGPA"].values[order], simple["y_pred"][order],
             color="#E74C3C", linewidth=3, label="Regression Line")
    _finish_plot("Simple Linear Regression: CGPA vs Salary",
                 "CGPA", "Salary", legend=True)

    # TASK 7: MULTIPLE LINEAR REGRESSION
    print("\n--- MULTIPLE LINEAR REGRESSION ---")
    multi = fit_multiple_regression(df)
    print("\nRegression Coefficients:")
    for feature, coefficient in zip(features, multi["model"].coef_):
        print(feature, "=", round(coefficient, 4))
    print("Intercept =", round(multi["model"].intercept_, 4))
    for k, v in multi["metrics"].items():
        print(k, "=", round(v, 4))

    # TASK 8: LINEAR ALGEBRA
    print("\n--- LINEAR ALGEBRA ---")
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    print("Matrix A:\n", A)
    print("\nMatrix B:\n", B)
    print("\nMatrix Addition:\n", A + B)
    print("\nMatrix Multiplication:\n", A @ B)
    print("\nDot Product:")
    print(np.dot(np.array([2, 3]), np.array([4, 5])))
    X_matrix = np.array([[1, 2], [1, 3], [1, 4]])
    y_matrix = np.array([5, 7, 9])
    beta = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ y_matrix
    print("\nRegression Beta:")
    print(beta)

    # TASK 9: GRADIENT DESCENT
    print("\n--- GRADIENT DESCENT ---")
    m, b, cost_history = gradient_descent(df)
    print("Final slope =", round(m, 4))
    print("Final intercept =", round(b, 4))
    print("Final cost =", round(cost_history[-1], 4))
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(len(cost_history)), y=cost_history,
                 color="#F39C12", linewidth=3)
    _finish_plot("Gradient Descent Cost Reduction", "Iteration", "Cost")

    # TASK 10: PLACEMENT PREDICTION
    print("\n--- LOGISTIC REGRESSION: PLACEMENT PREDICTION ---")
    place = fit_placement_model(df)
    for k, v in place["metrics"].items():
        print(f"{k:<10}:", round(v, 3))
    print("\nConfusion Matrix:")
    print(place["confusion"])

    # TASK 11: MODEL EVALUATION AND RESIDUAL ANALYSIS
    print("\n--- MODEL EVALUATION ---")
    y_test_eval = np.asarray(multi["y_test"]).reshape(-1)
    y_pred_eval = np.asarray(multi["y_pred"]).reshape(-1)
    if len(y_test_eval) != len(y_pred_eval):
        raise ValueError(
            f"Salary evaluation size mismatch: "
            f"actual={len(y_test_eval)}, predicted={len(y_pred_eval)}")
    for k, v in regression_metrics(y_test_eval, y_pred_eval).items():
        print(f"{k:<5}:", round(v, 3))

    residuals = y_test_eval - y_pred_eval
    print("\nResidual Analysis")
    print("Mean Residual    :", round(residuals.mean(), 3))
    print("Maximum Residual :", round(residuals.max(), 3))
    print("Minimum Residual :", round(residuals.min(), 3))

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred_eval, y=residuals, s=75, alpha=0.75,
                    color="#6C5CE7", edgecolor="white", linewidth=0.6)
    plt.axhline(0, color="#E74C3C", linestyle="--", linewidth=2.5,
                label="Zero Residual")
    _finish_plot("Residual Analysis", "Predicted Salary", "Residual",
                 legend=True)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test_eval, y=y_pred_eval, s=75, alpha=0.75,
                    color="#00B894", edgecolor="white", linewidth=0.6)
    lo = min(y_test_eval.min(), y_pred_eval.min())
    hi = max(y_test_eval.max(), y_pred_eval.max())
    plt.plot([lo, hi], [lo, hi], color="#E74C3C", linestyle="--",
             linewidth=2.5, label="Perfect Prediction")
    _finish_plot("Actual vs Predicted Salary", "Actual Salary",
                 "Predicted Salary", legend=True)

    print("\n--- FINAL PROJECT SUMMARY ---")
    print("Student Placement Analysis and Prediction completed successfully.")
    print("Best placement prediction model: Logistic Regression")
    print("Regression model evaluated using MAE, MSE, RMSE and R2.")
    print("Residual analysis completed.")

    print("\n======================================")
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("======================================")
    print("\nTasks completed:")
    for i, t in enumerate([
        "Data preprocessing", "Descriptive statistics",
        "Probability and Bayes theorem", "Hypothesis testing",
        "Correlation analysis", "Simple linear regression",
        "Multiple linear regression", "Linear algebra", "Gradient descent",
        "Placement prediction", "Model evaluation and residual analysis"], 1):
        print(f"{i}. {t}")


# ============================================================
# UNIPLACE AI - STREAMLIT WEB APP (streamlit run stu_place_final.py)
# ============================================================

INK, TEAL, BLUE, AMBER, ROSE, SLATE = (
    "#1F2A44", "#0E7C7B", "#3A6EA5", "#C98A1B", "#B5473F", "#5B6B7F")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&display=swap');

html, body, .stApp, .stMarkdown, button, input, textarea, label {
    font-family: 'Public Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top right, rgba(14,124,123,0.07), transparent 32%),
        linear-gradient(135deg, #F7F9FC 0%, #F1F4F8 100%);
}

#MainMenu, footer {
    visibility: hidden;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1320px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #F5F8FA 100%);
    border-right: 1px solid #E4E8EE;
}

[data-testid="stSidebarNav"]::before {
    content: "UniPlace AI";
    display: block;
    padding: 1.2rem 1rem .6rem;
    font-size: 1.35rem;
    font-weight: 700;
    color: #0E7C7B;
}

/* Page heading */
.ph h1 {
    font-size: 1.85rem;
    font-weight: 700;
    color: #172238;
    margin: 0;
    padding: 0;
    letter-spacing: -0.02em;
}

.ph p {
    color: #667085;
    margin: .25rem 0 1.5rem;
    font-size: .95rem;
}

/* KPI cards */
.kpi {
    background: rgba(255,255,255,0.95);
    border: 1px solid #E5E9F0;
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 5px 18px rgba(31,42,68,0.06);
    height: 100%;
    transition: all .2s ease;
}

.kpi:hover {
    transform: translateY(-2px);
    box-shadow: 0 9px 25px rgba(31,42,68,0.10);
}

.kl {
    font-size: .78rem;
    color: #667085;
    font-weight: 600;
}

.kv {
    font-size: 1.75rem;
    font-weight: 700;
    color: #172238;
    line-height: 1.35;
    margin-top: 4px;
}

.kn {
    font-size: .75rem;
    color: #98A2B3;
    margin-top: 3px;
}

/* Result cards */
.res {
    background: linear-gradient(135deg, #FFFFFF 0%, #F8FCFC 100%);
    border: 1px solid #E4E8EE;
    border-left: 6px solid #0E7C7B;
    border-radius: 16px;
    padding: 22px 26px;
    box-shadow: 0 6px 20px rgba(31,42,68,.07);
}

.res.warn {
    border-left-color: #C98A1B;
}

.res.risk {
    border-left-color: #B5473F;
}

.rl {
    font-size: .85rem;
    color: #667085;
    font-weight: 500;
}

.rv {
    font-size: 2.5rem;
    font-weight: 700;
    color: #172238;
    line-height: 1.2;
    margin: 4px 0;
}

.rs {
    font-size: 1rem;
    font-weight: 600;
    color: #0E7C7B;
}

.res.warn .rs {
    color: #C98A1B;
}

.res.risk .rs {
    color: #B5473F;
}

/* Streamlit containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,.92);
    border: 1px solid #E4E8EE;
    border-radius: 16px;
    box-shadow: 0 4px 16px rgba(31,42,68,.04);
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    border-radius: 10px;
    font-weight: 600;
    border: 1px solid #D9E0E7;
    transition: all .2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 5px 15px rgba(31,42,68,.10);
}

/* Primary button */
button[kind="primary"] {
    background: linear-gradient(135deg, #0E7C7B, #116F78) !important;
    border: none !important;
    box-shadow: 0 5px 15px rgba(14,124,123,.22);
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 600;
}

/* Info / warning boxes */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Selectboxes and inputs */
div[data-baseweb="select"] > div,
input {
    border-radius: 10px !important;
}
</style>
"""


def _style(fig, h=350):
    fig.update_layout(
        height=h,
        margin=dict(l=15, r=15, t=55, b=35),

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",

        font=dict(
            family="Public Sans, sans-serif",
            color=INK,
            size=12
        ),

        title=dict(
            font=dict(
                size=16,
                color="#172238"
            ),
            x=0,
            xanchor="left"
        ),

        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0,
            font=dict(size=11)
        ),

        colorway=[TEAL, BLUE, AMBER, ROSE, SLATE],

        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#DDE3EA",
            font=dict(
                family="Public Sans",
                size=12,
                color="#172238"
            )
        ),

        hovermode="closest"
    )

    fig.update_xaxes(
        gridcolor="#EEF1F5",
        zeroline=False,
        showline=False
    )

    fig.update_yaxes(
        gridcolor="#EEF1F5",
        zeroline=False,
        showline=False
    )

    return fig


def run_app():
    import streamlit as st
    import plotly.express as px

    st.set_page_config(page_title="UniPlace AI", page_icon=":material/school:",
                       layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    @st.cache_resource(show_spinner="Loading data and training models...")
    def load_everything():
        df = clean_data(load_data())
        simple = fit_simple_regression(df)
        multi = fit_multiple_regression(df)
        place = fit_placement_model(df)
        return df, simple, multi, place, support_table(df, place)

    try:
        df, simple, multi, place, sup = load_everything()
    except FileNotFoundError:
        st.error(f"Dataset not found. Put `student_placement.csv` next to "
                 f"`{Path(__file__).name}`.")
        st.stop()
    except Exception as e:
        st.error(f"Could not load the project data: {e}")
        st.stop()

    placed = df[df["Placement_Status"] == "Placed"]
    n_total = len(df)
    n_not = int((df["Placement_Status"] == "Not Placed").sum())
    rate = (n_total - n_not) / n_total * 100
    tiers = sup["Support_Priority"].value_counts()
    n_high, n_med = int(tiers.get("High", 0)), int(tiers.get("Medium", 0))
    pretty = lambda s: s.replace("_", " ")
    BAND = ["<6", "6-7", "7-8", "8-9", "9-10"]

    # ---------- helpers ----------
    def head(title, sub):
        st.markdown(f'<div class="ph"><h1>{title}</h1><p>{sub}</p></div>',
                    unsafe_allow_html=True)

    def kpi(col, label, value, note="&nbsp;"):
        col.markdown(f'<div class="kpi"><div class="kl">{label}</div>'
                     f'<div class="kv">{value}</div><div class="kn">{note}</div></div>',
                     unsafe_allow_html=True)

    chart_n = [0]

    def show(fig):
        chart_n[0] += 1
        with st.container(border=True):
            st.plotly_chart(fig, width="stretch", key=f"chart_{chart_n[0]}")

    def imbalance_note():
        st.caption(
            f"Note: only {n_not} of {n_total:,} students in the dataset are 'Not Placed', so the "
            "model's probabilities are high for almost everyone. Read them as relative "
            "readiness rather than absolute chances.")

    # ---------- charts ----------
    def ch_salary_hist():
        f = px.histogram(
        placed,
        x="Salary",
        nbins=18,
        marginal="box",
        title="Salary Distribution",
        color_discrete_sequence=[TEAL],
        opacity=0.88
    )

        f.update_traces(
        marker_line_color="white",
        marker_line_width=1
    )

        f.update_layout(
        xaxis_title="Salary (LPA)",
        yaxis_title="Students",
        bargap=0.08
    )

        return _style(f, 355)

    def ch_cgpa_band():
        d = (
        placed.assign(
            CGPA_Band=pd.cut(
                placed["CGPA"],
                bins=[0, 6, 7, 8, 9, 10],
                labels=["<6", "6-7", "7-8", "8-9", "9-10"],
                include_lowest=True
            )
        )
        .groupby("CGPA_Band", observed=False)["Salary"]
        .mean()
        .reset_index()
    )

        f = px.bar(
        d,
        x="CGPA_Band",
        y="Salary",
        title="Average Salary by CGPA Band",
        color="Salary",
        color_continuous_scale=[
            "#DCEFED",
            "#9BD4D0",
            "#0E7C7B"
        ]
    )

        f.update_traces(
        texttemplate="%{y:.2f} LPA",
        textposition="outside",
        cliponaxis=False
    )

        f.update_layout(
        xaxis_title="CGPA Band",
        yaxis_title="Average Salary (LPA)",
        coloraxis_showscale=False
    )

        return _style(f, 355)

    def ch_internship():
        d = (
        placed.groupby("Internship")["Salary"]
        .mean()
        .reindex(["No", "Yes"])
        .reset_index()
    )

        d.columns = ["Internship", "Average Salary"]

        f = px.bar(
        d,
        x="Internship",
        y="Average Salary",
        title="Internship Impact on Salary",
        color="Internship",
        color_discrete_map={
            "Yes": TEAL,
            "No": BLUE
        }
    )

        f.update_traces(
        texttemplate="%{y:.2f} LPA",
        textposition="outside",
        cliponaxis=False
    )

        f.update_layout(
        xaxis_title="Internship",
        yaxis_title="Average Salary (LPA)",
        showlegend=False
    )

        return _style(f, 355)

    def ch_tiers():
        d = (
        sup["Support_Priority"]
        .value_counts()
        .reindex(["High", "Medium", "Low"])
        .fillna(0)
        .reset_index()
    )

        d.columns = ["Priority", "Students"]

        total_support = int(
        d.loc[d["Priority"].isin(["High", "Medium"]), "Students"].sum()
    )

        f = px.pie(
        d,
        names="Priority",
        values="Students",
        hole=0.68,
        title="Student Support Priority",
        color="Priority",
        color_discrete_map={
            "High": ROSE,
            "Medium": AMBER,
            "Low": TEAL
        }
    )

        f.update_traces(
        textposition="outside",
        textinfo="label+percent",
        marker=dict(line=dict(color="white", width=3))
    )

        f.add_annotation(
        text=f"<b>{total_support}</b><br>Need support",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(
            size=16,
            color="#172238"
        )
    )

        return _style(f, 355)

    # ---------- pages ----------
    def page_dashboard():
        head("Dashboard", "UniPlace AI - University Placement Intelligence System")
        c = st.columns(5)
        kpi(c[0], "Total students", f"{n_total:,}", "after cleaning")
        kpi(c[1], "Placement rate", f"{rate:.1f}%", f"{n_total - n_not:,} placed")
        kpi(c[2], "Average salary", f"{placed['Salary'].mean():.2f} LPA", "placed students")
        kpi(c[3], "Highest salary", f"{placed['Salary'].max():.2f} LPA", "")
        kpi(c[4], "Need support", f"{n_high}", f"high priority, +{n_med} medium")
        st.write("")
        a, b = st.columns(2)
        with a: show(ch_salary_hist())
        with b: show(ch_cgpa_band())
        a, b = st.columns(2)
        with a: show(ch_internship())
        with b: show(ch_tiers())
        imbalance_note()

    def page_students():
        head("Student Analysis", "Browse, search and filter student records.")
        with st.container(border=True):
            a, b, c, d = st.columns([1.4, 1, 1, 1])
            q = a.text_input("Search Student ID", placeholder="e.g. STU0042")
            g = b.selectbox("Gender", ["All"] + sorted(df["Gender"].unique()))
            i = c.selectbox("Internship", ["All", "Yes", "No"])
            s = d.selectbox("Status", ["All", "Placed", "Not Placed"])
            e, f = st.columns([2, 2])
            cg = e.slider("CGPA range", 0.0, 10.0, (0.0, 10.0), 0.1)
            pr = f.multiselect("Support priority", ["High", "Medium", "Low"],
                               default=["High", "Medium", "Low"])
        t = sup[sup["CGPA"].between(*cg) & sup["Support_Priority"].isin(pr)]
        if q: t = t[t["Student_ID"].str.contains(q.strip(), case=False)]
        if g != "All": t = t[t["Gender"] == g]
        if i != "All": t = t[t["Internship"] == i]
        if s != "All": t = t[t["Placement_Status"] == s]
        st.caption(f"Showing {len(t):,} of {n_total:,} students")
        if t.empty:
            st.info("No students match these filters. Try widening the CGPA range or clearing a filter.")
            return
        cols = ["Student_ID", "Gender", "CGPA", "Attendance", "Programming_Score",
                "Communication_Score", "Aptitude_Score", "Internship", "Projects",
                "Backlogs", "Placement_Status", "Salary", "Placement_Probability",
                "Support_Priority"]
        st.dataframe(
            t[cols], hide_index=True, width="stretch", height=480,
            column_config={
                "Attendance": st.column_config.ProgressColumn(
                    "Attendance", min_value=0, max_value=100, format="%.0f%%"),
                "Placement_Probability": st.column_config.ProgressColumn(
                    "Placement prob.", min_value=0, max_value=100, format="%.1f%%"),
                "Salary": st.column_config.NumberColumn("Salary (LPA)", format="%.2f"),
                "Programming_Score": "Programming", "Communication_Score": "Communication",
                "Aptitude_Score": "Aptitude", "Placement_Status": "Status",
                "Support_Priority": "Support"})
        st.download_button("Download filtered records (CSV)", t[cols].to_csv(index=False),
                           "filtered_students.csv", "text/csv", key="dl_students")

    def predict_row(row):
        prob = float(placement_probability(place, row)[0]) * 100
        sal = float(multi["model"].predict(row[features])[0])
        return prob, sal

    def result_card(prob, sal, row):
        tier = str(tier_series([prob])[0])
        ok = prob >= 50
        cls = "res" if (ok and tier == "Low") else ("res warn" if ok else "res risk")
        label = "Likely to be placed" if ok else "Unlikely to be placed"
        flags = focus_flags(row).iloc[0]
        areas = ", ".join(flags.index[flags.values]) or "None flagged"
        st.markdown(
            f'<div class="{cls}"><div class="rl">Predicted placement probability</div>'
            f'<div class="rv">{prob:.1f}%</div><div class="rs">{label}</div></div>',
            unsafe_allow_html=True)
        st.write("")
        c = st.columns(3)
        kpi(c[0], "Support priority", tier, "relative to cohort")
        kpi(c[1], "Estimated package", f"{max(sal, 0):.2f} LPA" if ok else "-", "if placed")
        kpi(c[2], "Cohort standing", f"{(sup['Placement_Probability'] < prob).mean() * 100:.0f}th pct",
            "placement probability")
        st.write("")
        st.markdown(f"**Focus areas:** {areas}")

    def page_predict():
        head("Placement Prediction", "Enter a student profile to get a placement prediction.")
        with st.form("predict_form", border=True):
            c1, c2, c3 = st.columns(3)
            cgpa = c1.number_input("CGPA", 0.0, 10.0, 7.7, 0.01)
            att = c2.number_input("Attendance (%)", 0.0, 100.0, 82.0, 0.1)
            sh = c3.number_input("Study hours / day", 0.0, 16.0, 3.8, 0.1)
            prog = c1.number_input("Programming score", 0.0, 100.0, 95.0, 0.1)
            comm = c2.number_input("Communication score", 0.0, 100.0, 80.0, 0.1)
            apt = c3.number_input("Aptitude score", 0.0, 100.0, 85.0, 0.1)
            intern = c1.selectbox("Internship", ["Yes", "No"])
            proj = c2.number_input("Projects", 0, 15, 3)
            back = c3.number_input("Backlogs", 0, 15, 0)
            submit = st.form_submit_button("Predict placement", type="primary")
        if not submit:
            st.info("Fill in the details above and press **Predict placement**.")
            return
        row = pd.DataFrame([{
            "CGPA": cgpa, "Attendance": att, "Study_Hours": sh, "Programming_Score": prog,
            "Communication_Score": comm, "Aptitude_Score": apt, "Internship": intern,
            "Projects": proj, "Backlogs": back}])
        try:
            prob, sal = predict_row(row)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return
        result_card(prob, sal, row)
        z = place["scaler"].transform(_encode_placement_X(row))[0]
        d = pd.DataFrame({"Factor": [pretty(c) for c in placement_features],
                          "Effect": place["model"].coef_[0] * z})
        d = d.sort_values("Effect")
        f = px.bar(d, x="Effect", y="Factor", orientation="h",
                   title="What moved this prediction (vs. an average student)")
        f.update_traces(marker_color=[TEAL if v >= 0 else ROSE for v in d["Effect"]])
        f.update_layout(xaxis_title="Effect on log-odds of placement", yaxis_title="")
        show(_style(f, 340))
        imbalance_note()

    def page_salary():
        head("Salary Analysis", "Salary statistics and regression performance (placed students).")
        s = placed["Salary"]
        m = multi["metrics"]
        c = st.columns(6)
        kpi(c[0], "Mean", f"{s.mean():.2f}", "LPA")
        kpi(c[1], "Median", f"{s.median():.2f}", "LPA")
        kpi(c[2], "Std. deviation", f"{s.std():.2f}", "LPA")
        kpi(c[3], "R2 (test)", f"{m['R2']:.3f}", "multiple regression")
        kpi(c[4], "MAE", f"{m['MAE']:.2f}", "LPA")
        kpi(c[5], "RMSE", f"{m['RMSE']:.2f}", "LPA")
        st.write("")
        t1, t2, t3, t4 = st.tabs(["Actual vs predicted", "Residuals", "CGPA regression",
                                  "Coefficients"])
        yt, yp = np.asarray(multi["y_test"]), np.asarray(multi["y_pred"])
        with t1:
            f = px.scatter(x=yt, y=yp, color_discrete_sequence=[TEAL], opacity=0.75,
                           title="Actual vs predicted salary")
            lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
            f.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                          line=dict(color=ROSE, dash="dash"))
            f.update_layout(xaxis_title="Actual (LPA)", yaxis_title="Predicted (LPA)",
                            showlegend=False)
            show(_style(f, 400))
        with t2:
            f = px.scatter(x=yp, y=yt - yp, color_discrete_sequence=[BLUE], opacity=0.75,
                           title="Residual analysis")
            f.add_hline(y=0, line_dash="dash", line_color=ROSE)
            f.update_layout(xaxis_title="Predicted (LPA)", yaxis_title="Residual")
            show(_style(f, 400))
            r = yt - yp
            st.caption(f"Mean residual {r.mean():.3f} | max {r.max():.3f} | min {r.min():.3f}")
        with t3:
            X, y, p = simple["X_test"]["CGPA"].to_numpy(), np.asarray(simple["y_test"]), simple["y_pred"]
            o = np.argsort(X)
            f = px.scatter(x=X, y=y, color_discrete_sequence=[BLUE], opacity=0.75,
                           title="CGPA vs salary (simple linear regression)")
            f.add_scatter(x=X[o], y=p[o], mode="lines", name="Regression line",
                          line=dict(color=ROSE, width=3))
            f.update_layout(xaxis_title="CGPA", yaxis_title="Salary (LPA)", showlegend=False)
            show(_style(f, 400))
            sm = simple["metrics"]
            st.caption(f"Slope {simple['model'].coef_[0]:.3f} | intercept "
                       f"{simple['model'].intercept_:.3f} | R2 {sm['R2']:.3f} | RMSE {sm['RMSE']:.3f}")
        with t4:
            d = pd.DataFrame({"Factor": [pretty(c) for c in features],
                              "Coefficient": multi["model"].coef_}).sort_values("Coefficient")
            f = px.bar(d, x="Coefficient", y="Factor", orientation="h",
                       title="Multiple regression coefficients (LPA per unit)")
            f.update_traces(marker_color=[TEAL if v >= 0 else ROSE for v in d["Coefficient"]])
            f.update_layout(yaxis_title="")
            show(_style(f, 400))
        t, p, a, b = internship_ttest(df)
        st.info(f"**Internship effect:** placed students with an internship average {a:.2f} LPA vs "
                f"{b:.2f} LPA without (Welch t = {t:.2f}, p = {p:.3f}) - "
                f"{'a statistically significant' if p < 0.05 else 'no statistically significant'} difference.")

    def page_factors():
        head("Placement Factors", "Which attributes move placement and salary.")
        pear, spear = correlations(df)
        kind = st.radio("Correlation method", ["Pearson", "Spearman"], horizontal=True)
        corr = (pear if kind == "Pearson" else spear)
        a, b = st.columns([1.3, 1])
        with a:
            m = corr.rename(index=pretty, columns=pretty).round(2)
            f = px.imshow(m, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                          aspect="auto", title=f"{kind} correlation matrix")
            show(_style(f, 460))
        with b:
            d = corr["Salary"].drop("Salary").sort_values()
            d.index = [pretty(i) for i in d.index]
            f = px.bar(x=d.values, y=d.index, orientation="h", title="Correlation with salary")
            f.update_traces(marker_color=[TEAL if v >= 0 else ROSE for v in d.values])
            f.update_layout(xaxis_title="Correlation", yaxis_title="")
            show(_style(f, 460))
        coef = pd.Series(place["model"].coef_[0], index=[pretty(c) for c in placement_features]).sort_values()
        f = px.bar(x=coef.values, y=coef.index, orientation="h",
                   title="Placement model: standardized logistic coefficients")
        f.update_traces(marker_color=[TEAL if v >= 0 else ROSE for v in coef.values])
        f.update_layout(xaxis_title="Coefficient", yaxis_title="")
        show(_style(f, 360))
        ps = probability_stats(df)
        c = st.columns(4)
        kpi(c[0], "P(Placement)", f"{ps['p_placed']:.3f}")
        kpi(c[1], "P(CGPA > 8)", f"{ps['p_cgpa_8']:.3f}")
        kpi(c[2], "P(Placement | CGPA > 8)", f"{ps['p_placed_given_cgpa']:.3f}")
        kpi(c[3], "Bayes check", f"{ps['bayes']:.3f}", "same quantity via Bayes")
        st.caption(f"Placement model test set had only {place['test_not_placed']} 'Not Placed' student(s), "
                   "so classification metrics are not very informative; coefficient signs can be noisy.")

    def page_whatif():
        head("What-If Analysis", "Change a profile and see how the predicted probability moves.")
        ids = ["Average student"] + list(df["Student_ID"])
        base_id = st.selectbox("Start from", ids)
        if base_id == "Average student":
            base = {c: float(df[c].median()) for c in placement_features if c != "Internship"}
            base["Internship"] = df["Internship"].mode()[0]
            base["Projects"], base["Backlogs"] = int(round(base["Projects"])), int(round(base["Backlogs"]))
        else:
            base = df[df["Student_ID"] == base_id].iloc[0][placement_features].to_dict()
        k = base_id
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            new = {
                "CGPA": c1.slider("CGPA", 0.0, 10.0, float(base["CGPA"]), 0.01, key=f"c{k}"),
                "Attendance": c2.slider("Attendance (%)", 0.0, 100.0, float(base["Attendance"]), 0.1, key=f"a{k}"),
                "Study_Hours": c3.slider("Study hours / day", 0.0, 12.0, float(base["Study_Hours"]), 0.1, key=f"s{k}"),
                "Programming_Score": c1.slider("Programming score", 0.0, 100.0, float(base["Programming_Score"]), 0.1, key=f"p{k}"),
                "Communication_Score": c2.slider("Communication score", 0.0, 100.0, float(base["Communication_Score"]), 0.1, key=f"m{k}"),
                "Aptitude_Score": c3.slider("Aptitude score", 0.0, 100.0, float(base["Aptitude_Score"]), 0.1, key=f"t{k}"),
                "Internship": c1.selectbox("Internship", ["Yes", "No"], index=0 if base["Internship"] == "Yes" else 1, key=f"i{k}"),
                "Projects": c2.slider("Projects", 0, 10, int(base["Projects"]), key=f"j{k}"),
                "Backlogs": c3.slider("Backlogs", 0, 8, int(base["Backlogs"]), key=f"b{k}"),
            }
        r0, r1 = pd.DataFrame([base]), pd.DataFrame([new])
        (p0, s0), (p1, s1) = predict_row(r0), predict_row(r1)
        c = st.columns(3)
        kpi(c[0], "Baseline probability", f"{p0:.1f}%", f"~{max(s0, 0):.2f} LPA")
        kpi(c[1], "Scenario probability", f"{p1:.1f}%", f"~{max(s1, 0):.2f} LPA")
        kpi(c[2], "Change", f"{p1 - p0:+.1f} pts", f"{s1 - s0:+.2f} LPA estimated salary")
        imbalance_note()

    def page_risk():
        head("Risk / Support Analysis",
             "Students who may benefit from extra placement preparation.")
        st.info("This is a **support priority indicator**, not a guaranteed prediction. It ranks "
                f"students by the placement model's probability (High below {HIGH_CUT}%, Medium "
                f"below {MED_CUT}%, plus anyone not yet placed) and lists likely focus areas.")
        c = st.columns(3)
        kpi(c[0], "High priority", n_high)
        kpi(c[1], "Medium priority", n_med)
        kpi(c[2], "Low priority", int(tiers.get("Low", 0)))
        st.write("")
        pr = st.multiselect("Show priority", ["High", "Medium"], default=["High"])
        t = sup[sup["Support_Priority"].isin(pr)].sort_values("Placement_Probability")
        if t.empty:
            st.info("No students in the selected priority groups.")
            return
        a, b = st.columns([1.6, 1])
        with a:
            st.dataframe(
                t[["Student_ID", "CGPA", "Backlogs", "Placement_Status", "Placement_Probability",
                   "Support_Priority", "Focus_Areas"]],
                hide_index=True, width="stretch", height=430,
                column_config={"Placement_Probability": st.column_config.NumberColumn(
                    "Prob. (%)", format="%.1f"), "Placement_Status": "Status",
                    "Support_Priority": "Priority", "Focus_Areas": "Focus areas"})
        with b:
            counts = (t["Focus_Areas"].str.split(", ").explode().loc[lambda s: s != "-"]
                      .value_counts().sort_values())
            f = px.bar(x=counts.values, y=counts.index, orientation="h",
                       title="Most common focus areas", color_discrete_sequence=[AMBER])
            f.update_layout(xaxis_title="Students", yaxis_title="")
            show(_style(f, 430))
        st.download_button("Download support list (CSV)", t.to_csv(index=False),
                           "support_priority_students.csv", "text/csv", key="dl_support")

    def report_md():
        t, p, a, b = internship_ttest(df)
        pm, mm = place["metrics"], multi["metrics"]
        return (
            f"# UniPlace AI - Placement Summary\n\n"
            f"- Students analysed: {n_total:,}\n- Placement rate: {rate:.1f}% ({n_total - n_not:,} placed, {n_not} not placed)\n"
            f"- Salary (placed): mean {placed['Salary'].mean():.2f}, median {placed['Salary'].median():.2f}, "
            f"max {placed['Salary'].max():.2f} LPA\n"
            f"- Internship: {a:.2f} LPA vs {b:.2f} LPA without (Welch p = {p:.3f})\n"
            f"- Salary model: R2 {mm['R2']:.3f}, MAE {mm['MAE']:.2f}, RMSE {mm['RMSE']:.2f}\n"
            f"- Placement model: accuracy {pm['Accuracy']:.3f}, F1 {pm['F1 Score']:.3f}\n"
            f"- Support priority: {n_high} high, {n_med} medium\n")

    def page_reports():
        head("Reports", "Presentation-ready placement summary.")
        t, p, a, b = internship_ttest(df)
        mm = multi["metrics"]
        with st.container(border=True):
            st.markdown(
                f"**Overview.** {n_total:,} students analysed, {rate:.1f}% placed "
                f"({n_not} not placed). Placed students earn {placed['Salary'].mean():.2f} LPA on average "
                f"(median {placed['Salary'].median():.2f}, highest {placed['Salary'].max():.2f}).\n\n"
                f"**Internships.** {a:.2f} LPA with an internship vs {b:.2f} LPA without "
                f"(p = {p:.3f}, {'significant' if p < 0.05 else 'not significant'}).\n\n"
                f"**Models.** Salary regression R2 = {mm['R2']:.2f} (RMSE {mm['RMSE']:.2f} LPA). "
                f"Placement logistic model accuracy = {place['metrics']['Accuracy']:.2f}.\n\n"
                f"**Support.** {n_high} students are high priority and {n_med} medium for additional preparation.")
        a_, b_ = st.columns(2)
        with a_: show(ch_salary_hist())
        with b_: show(ch_cgpa_band())
        a_, b_ = st.columns(2)
        with a_: show(ch_internship())
        with b_: show(ch_tiers())
        st.download_button("Download summary (Markdown)", report_md(), "uniplace_summary.md",
                           "text/markdown", key="dl_report")

    pages = [
        st.Page(page_dashboard, title="Dashboard", icon=":material/dashboard:", default=True),
        st.Page(page_students, title="Student Analysis", icon=":material/groups:"),
        st.Page(page_predict, title="Placement Prediction", icon=":material/psychology:"),
        st.Page(page_salary, title="Salary Analysis", icon=":material/payments:"),
        st.Page(page_factors, title="Placement Factors", icon=":material/insights:"),
        st.Page(page_whatif, title="What-If Analysis", icon=":material/tune:"),
        st.Page(page_risk, title="Risk / Support", icon=":material/health_and_safety:"),
        st.Page(page_reports, title="Reports", icon=":material/description:"),
    ]
    st.navigation(pages).run()


def _in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _in_streamlit():
        run_app()
    else:
        run_terminal()