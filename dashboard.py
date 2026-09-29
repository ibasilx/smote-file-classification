"""Local dashboard for the supplied model evaluation results."""
from pathlib import Path
import pandas as pd
import streamlit as st

RESULTS = Path(__file__).parent / "results" / "metrics.csv"
EXPERIMENTS = {"baseline": "Baseline", "smote": "SMOTE", "feature_selection": "SMOTE + feature selection"}
METRICS = {"accuracy": "Accuracy", "precision_macro": "Macro precision", "recall_macro": "Macro recall", "f1_macro": "Macro F1", "fpr_macro": "Macro false-positive rate"}

st.set_page_config(page_title="SMOTE File Classification", page_icon="📊", layout="wide")
st.title("File Classification Using SMOTE and Machine Learning")
st.caption("Academic / portfolio project · Evaluation of the supplied feature dataset")
if not RESULTS.is_file():
    st.error("Missing results/metrics.csv")
    st.stop()
results = pd.read_csv(RESULTS)
if results.empty or not {"experiment", "model", *METRICS}.issubset(results.columns):
    st.error("The results file does not have the expected columns.")
    st.stop()

cols = st.columns(3)
cols[0].metric("Samples", "47,482")
cols[1].metric("Classes", "20")
cols[2].metric("Input features", "512")
st.info("The repository does not include the process that generated the dataset's feature columns.")

st.subheader("Compare experiments")
experiment = st.selectbox("Experiment", list(EXPERIMENTS), format_func=lambda x: EXPERIMENTS[x])
metric = st.selectbox("Metric", list(METRICS), index=3, format_func=lambda x: METRICS[x])
view = results.loc[results.experiment == experiment, ["model", metric]].set_index("model")
view[metric] = view[metric] * 100
st.bar_chart(view, y=metric)
st.caption("Values are percentages from the recorded held-out test results; macro metrics weight each class equally.")

st.subheader("Recorded metrics")
selected = results.loc[results.experiment == experiment].copy()
selected["experiment"] = selected["experiment"].map(EXPERIMENTS)
for name in METRICS:
    selected[name] = selected[name] * 100
selected = selected.rename(columns={"experiment": "Experiment", "model": "Model", **METRICS})
st.dataframe(selected, hide_index=True, use_container_width=True, column_config={name: st.column_config.NumberColumn(format="%.2f%%") for name in METRICS.values()})

st.subheader("Method notes")
st.markdown("""
- Stratified 80/20 train/test split, random_state=42.
- SMOTE was applied to the training split only.
- The feature-selection experiment applies MinMaxScaler and SelectKBest with chi-squared scoring and k=256.
- No arbitrary raw-file upload or prediction function is included.
- The CSV filename is sift_512.csv; the feature-generation process is not present in the supplied code.
""")
