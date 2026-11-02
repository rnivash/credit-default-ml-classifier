import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

st.set_page_config(page_title="Credit Default Predictor", page_icon="💳", layout="wide")

MODEL_FILES = {
    'Logistic Regression': 'model/logistic_regression.joblib',
    'Decision Tree': 'model/decision_tree.joblib',
    'KNN': 'model/knn.joblib',
    'Naive Bayes': 'model/naive_bayes.joblib',
    'Support Vector Machine': 'model/support_vector_machine.joblib',
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_resource
def load_scaler():
    return joblib.load('model/scaler.joblib')

st.title("💳 Credit Card Default Prediction")
st.caption("Compare 5 pre-trained classifiers on your test data. No retraining happens here.")

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    choice = st.selectbox("Select model", list(MODEL_FILES.keys()))
    st.info("CSV must contain all 23 feature columns plus a 'target' column.")

if uploaded is None:
    st.write("👈 Upload `test_data.csv` from the sidebar to begin.")
    st.stop()

data = pd.read_csv(uploaded)
if 'target' not in data.columns:
    st.error("The CSV must contain a 'target' column.")
    st.stop()

X = data.drop(columns=['target'])
y = data['target']

scaler = load_scaler()
try:
    X = X[list(scaler.feature_names_in_)]
except KeyError as e:
    st.error(f"Missing required columns: {e}")
    st.stop()

model = load_model(MODEL_FILES[choice])
Xs = scaler.transform(X)
pred = model.predict(Xs)
prob = model.predict_proba(Xs)[:, 1]

st.subheader(f"Results: {choice}")
c = st.columns(6)
c[0].metric("Accuracy", f"{accuracy_score(y, pred):.4f}")
c[1].metric("AUC", f"{roc_auc_score(y, prob):.4f}")
c[2].metric("Precision", f"{precision_score(y, pred, zero_division=0):.4f}")
c[3].metric("Recall", f"{recall_score(y, pred):.4f}")
c[4].metric("F1", f"{f1_score(y, pred):.4f}")
c[5].metric("MCC", f"{matthews_corrcoef(y, pred):.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion Matrix**")
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(confusion_matrix(y, pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=['No default', 'Default'],
                yticklabels=['No default', 'Default'], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.markdown("**Classification Report**")
    rep = classification_report(y, pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).transpose().round(3))

with st.expander("Preview uploaded data"):
    st.dataframe(data.head(20))