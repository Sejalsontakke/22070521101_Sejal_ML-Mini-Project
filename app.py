# app.py
import warnings
warnings.filterwarnings("ignore")

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from statsmodels.tsa.arima.model import ARIMA
import streamlit as st

# ---------------- UI HEADER ----------------
st.set_page_config(page_title="California Wildfire Analysis", layout="wide")
sns.set_style("whitegrid")

st.title("🔥 California Wildfire Analysis Dashboard")
st.caption("Upload your **cleaned** dataset (CSV). The app will run EDA, ML classification, clustering, and a simple ARIMA forecast.")

st.subheader("Upload the cleaned dataset (.csv)")
uploaded_file = st.file_uploader("Drag and drop file here", type="csv")

# ------------- SMALL HELPERS -------------
REQ_COLS = [
    "GIS Calculated Acres", "Cause", "Alarm Date", "Containment Date", "Year"
]

def check_required_columns(df: pd.DataFrame, required):
    missing = [c for c in required if c not in df.columns]
    return missing

def plot_to_streamlit(fig, use_container_width=True):
    st.pyplot(fig, use_container_width=use_container_width)
    plt.close(fig)

# ------------- APP MAIN -------------
if uploaded_file is None:
    st.info("Please upload the **cleaned** dataset to proceed.")
    st.stop()

# Load data
df = pd.read_csv(uploaded_file, low_memory=False)

missing = check_required_columns(df, REQ_COLS)
if missing:
    st.error(f"Your file is missing required column(s): {missing}. "
             "Please upload the cleaned CSV with these columns.")
    st.stop()

st.success("✅ Dataset loaded successfully!")
st.write("### 🔎 Preview")
st.dataframe(df.head(), use_container_width=True)
st.write("**Shape:**", df.shape)

# ---- Parse dates & create duration feature (idempotent) ----
for col in ["Alarm Date", "Containment Date"]:
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors="coerce")

df["Fire_Duration_Days"] = (
    (df["Containment Date"] - df["Alarm Date"]).dt.total_seconds() / (60 * 60 * 24)
)

# Clean minimal issues for analysis
df = df.dropna(subset=["GIS Calculated Acres"])
df = df[df["GIS Calculated Acres"] > 0]
# keep non-negative durations only (ignore NaN durations—will be handled later)
df = df[(df["Fire_Duration_Days"].isna()) | (df["Fire_Duration_Days"] >= 0)]

# Fill Cause missing with 'Unknown'
df["Cause"] = df["Cause"].fillna("Unknown")

# Large fire label (top 25%)
acre_threshold = df["GIS Calculated Acres"].quantile(0.75)
df["Large_Fire"] = (df["GIS Calculated Acres"] > acre_threshold).astype(int)

# Simplify causes (top-5 + Other)
top_causes = df["Cause"].value_counts().head(5).index.tolist()
df["Cause_Simplified"] = df["Cause"].apply(lambda x: x if x in top_causes else "Other")

# =========================
# Section: EDA
# =========================
st.header("1) Exploratory Data Analysis")

with st.expander("📊 Descriptive Statistics", expanded=True):
    num_cols = ["GIS Calculated Acres", "Fire_Duration_Days", "Year"]
    show_cols = [c for c in num_cols if c in df.columns]
    st.dataframe(df[show_cols].describe().T, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Distribution: Log(GIS Calculated Acres + 1)**")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(np.log1p(df["GIS Calculated Acres"]), kde=True, bins=50, ax=ax)
    ax.set_xlabel("Log(Acres + 1)")
    plot_to_streamlit(fig)

with col2:
    st.markdown("**Scatter: Log(Duration + 1) vs Log(Acres + 1)**")
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.log1p(df["Fire_Duration_Days"].fillna(0))
    y = np.log1p(df["GIS Calculated Acres"])
    sns.scatterplot(x=x, y=y, alpha=0.5, ax=ax)
    ax.set_xlabel("Log(Fire_Duration_Days + 1)")
    ax.set_ylabel("Log(GIS Calculated Acres + 1)")
    plot_to_streamlit(fig)

st.markdown("**Counts by Cause (Simplified)**")
fig, ax = plt.subplots(figsize=(7, 4))
order = df["Cause_Simplified"].value_counts().index
sns.countplot(y="Cause_Simplified", data=df, order=order, ax=ax)
plot_to_streamlit(fig)

# =========================
# Section: ML Classification
# =========================
st.header("2) Classification: Predict Large Fires (75th percentile)")

# Prepare features
REGRESSION_TARGET_LOG = "GIS Calculated Acres_log"
df[REGRESSION_TARGET_LOG] = np.log1p(df["GIS Calculated Acres"])

EXCLUDE_COLS = [
    "OBJECTID", "Fire Name", "Local Incident Number", "Alarm Date", "Containment Date",
    "Comments", "Complex Name", "IRWIN ID", "Fire Number (historical use)", "Complex ID",
    "DECADES", "GIS Calculated Acres", "Cause", "State", "Agency", "Unit ID",
    REGRESSION_TARGET_LOG, "Large_Fire"
]

X = df.drop(columns=EXCLUDE_COLS, errors="ignore")
# Ensure we include Cause_Simplified
if "Cause_Simplified" not in X.columns:
    X["Cause_Simplified"] = df["Cause_Simplified"]

# Separate targets
y_cls = df["Large_Fire"]

# Split numeric/categorical
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = ["Cause_Simplified"] if "Cause_Simplified" in X.columns else []

# Preprocess
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ],
    remainder="drop"
)

X_processed = preprocessor.fit_transform(X)
feature_names = numeric_features + (
    list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features))
    if categorical_features else []
)
X_final = pd.DataFrame(X_processed, columns=feature_names).fillna(0)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_cls, test_size=0.2, random_state=42
)

# Models
models = {
    "Logistic Regression": LogisticRegression(random_state=42, solver="liblinear"),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
}

def evaluate_model(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba[:, 1]) if y_proba is not None else np.nan
    return acc, prec, rec, f1, auc

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    yp = model.predict(X_test)
    ypp = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    acc, prec, rec, f1, auc = evaluate_model(y_test, yp, ypp)
    rows.append({
        "Model":
