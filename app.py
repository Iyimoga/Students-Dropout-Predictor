"""
Streamlit App: Student Dropout Predictor with Model Comparison
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import joblib
from PIL import Image

st.set_page_config(page_title="Student Dropout Predictor",
                   page_icon="🎓", layout="wide")

# ---------- load artifacts ----------
@st.cache_resource
def load_artifacts():
    return {
        "all_models": joblib.load("all_models.pkl"),
        "best_name": joblib.load("best_model_name.pkl"),
        "scaler": joblib.load("scaler.pkl"),
        "features": joblib.load("features.pkl"),
        "stats": joblib.load("feat_stats.pkl"),
        "comparison": joblib.load("comparison_df.pkl"),
    }

art = load_artifacts()
all_models = art["all_models"]
best_name = art["best_name"]
scaler = art["scaler"]
features = art["features"]
stats = art["stats"]
comparison = art["comparison"]

st.title("🎓 Student Dropout Risk Predictor")
st.caption(
    "Undergraduate Project · Model Comparison of Logistic Regression, "
    "Random Forest, and Support Vector Machine"
)

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Comparison", "ℹ️ About"])

# ============================================================
# TAB 1 — PREDICTION
# ============================================================
with tab1:
    st.subheader("Enter student information")

    model_choice = st.selectbox(
        "Choose a model for prediction",
        list(all_models.keys()),
        index=list(all_models.keys()).index(best_name),
        help=f"Default is the best-performing model ({best_name}).",
    )

    # split inputs into two columns to avoid a huge sidebar
    user_input = {}
    cols = st.columns(2)
    for i, feat in enumerate(features):
        lo = float(stats[feat]["min"])
        hi = float(stats[feat]["max"])
        med = float(stats[feat]["median"])
        with cols[i % 2]:
            if hi - lo > 1 and (hi == int(hi) and lo == int(lo)):
                user_input[feat] = st.number_input(
                    feat, min_value=lo, max_value=hi, value=med, step=1.0)
            else:
                user_input[feat] = st.number_input(
                    feat, min_value=lo, max_value=hi, value=med)

    if st.button("Predict Dropout Risk", type="primary", use_container_width=True):
        model = all_models[model_choice]
        X = pd.DataFrame([user_input])[features]
        X_scaled = scaler.transform(X)
        proba = model.predict_proba(X_scaled)[0, 1]
        pred = int(proba >= 0.5)

        c1, c2, c3 = st.columns(3)
        c1.metric("Model Used", model_choice)
        c2.metric("Dropout Probability", f"{proba*100:.1f}%")
        c3.metric("Prediction",
                  "⚠️ At Risk" if pred else "✅ Likely to Graduate")

        st.progress(float(proba))
        if proba >= 0.7:
            st.error("**High dropout risk** — recommend immediate academic counselling.")
        elif proba >= 0.4:
            st.warning("**Moderate risk** — monitor and offer support services.")
        else:
            st.success("**Low risk** — student is on track.")

# ============================================================
# TAB 2 — MODEL COMPARISON
# ============================================================
with tab2:
    st.subheader("Performance of all three models on the test set")
    st.dataframe(
        comparison.style.highlight_max(axis=0, color='#c7e9c0')
                        .format("{:.4f}"),
        use_container_width=True,
    )
    st.caption(f"✅ Best model by ROC-AUC: **{best_name}**")

    st.markdown("### 📈 Visual comparisons")
    try:
        st.image("comparison_bars.png", caption="Metric comparison across models")
        st.image("roc_curves.png", caption="ROC curves")
        st.image("confusion_matrices.png", caption="Confusion matrices")
        st.image("mi_scores.png", caption="Top features by Mutual Information")
    except Exception:
        st.info("Run `python train.py` first to generate the comparison plots.")

    st.markdown("### 📝 Interpretation")
    st.markdown(
        """
        - **Logistic Regression** — a strong linear baseline. Fast, fully
          interpretable (coefficients = feature effects), but cannot model
          non-linear interactions.
        - **Random Forest** — bagging ensemble of decision trees. Handles
          non-linearities and feature interactions well and typically achieves
          the best balance of precision and recall on tabular data.
        - **Support Vector Machine (RBF kernel)** — margin-based classifier
          that maps features into a higher-dimensional space. Competitive
          accuracy but less interpretable and slower to train.
        """
    )

# ============================================================
# TAB 3 — ABOUT
# ============================================================
with tab3:
    st.markdown(
        """
        ### About this project
        This application predicts the likelihood of a student dropping out of
        university based on demographic, socio-economic, and academic-performance
        features.

        **Pipeline**
        1. Data cleaning (duplicates, missing values, outlier trimming)
        2. Feature selection using Mutual Information (`MI > 0.01`)
        3. Standard scaling of numeric features
        4. Training three classifiers and comparing them on 5-fold CV and held-out test set
        5. Deployment with Streamlit

        **Metrics reported**: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
        """
    )
