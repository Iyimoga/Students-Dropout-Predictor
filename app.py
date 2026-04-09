"""
Streamlit App: Student Dropout Predictor (Nigerian Context)
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import joblib
from nigerian_labels import (
    YES_NO, GENDER, MARITAL_STATUS, APPLICATION_MODE,
    PREVIOUS_QUALIFICATION, COURSE, PARENT_QUALIFICATION,
    PARENT_OCCUPATION, percentage_to_portuguese, nigerian_grade_label,
)

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
        "comparison": joblib.load("comparison_df.pkl"),
    }

art = load_artifacts()
all_models = art["all_models"]
best_name = art["best_name"]
scaler = art["scaler"]
features = art["features"]
comparison = art["comparison"]

st.title("🎓 Student Dropout Risk Predictor")
st.caption(
    "Predicting dropout risk for Nigerian university students · "
    "Undergraduate Final Year Project"
)

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Comparison", "ℹ️ About"])

# ============================================================
# TAB 1 — PREDICTION (grouped sections)
# ============================================================
with tab1:
    model_choice = st.selectbox(
        "Select prediction model",
        list(all_models.keys()),
        index=list(all_models.keys()).index(best_name),
        help=f"Default is the best-performing model: {best_name}",
    )

    user_input = {}

    # ------------------------------------------------------------------
    # 📇 SECTION 1 — PERSONAL INFORMATION
    # ------------------------------------------------------------------
    st.markdown("### 📇 1. Personal Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        user_input["Gender"] = GENDER[
            st.selectbox("Gender", list(GENDER.keys()))]
    with c2:
        user_input["Age at enrollment"] = st.number_input(
            "Age at enrollment", min_value=15, max_value=60, value=18, step=1,
            help="Your age (in years) on the day you were admitted."
        )
    with c3:
        user_input["Marital status"] = MARITAL_STATUS[
            st.selectbox("Marital status", list(MARITAL_STATUS.keys()))]

    user_input["Displaced"] = YES_NO[
        st.radio(
            "Did you relocate from your hometown to attend this university?",
            list(YES_NO.keys()), horizontal=True,
            help="Yes if you moved to a different state/city for school."
        )]

    # ------------------------------------------------------------------
    # 🎓 SECTION 2 — EDUCATIONAL BACKGROUND
    # ------------------------------------------------------------------
    st.markdown("### 🎓 2. Educational Background")
    c1, c2 = st.columns(2)
    with c1:
        user_input["Previous qualification"] = PREVIOUS_QUALIFICATION[
            st.selectbox(
                "Highest qualification before admission",
                list(PREVIOUS_QUALIFICATION.keys()),
                help="What you had before entering this university."
            )]
    with c2:
        user_input["Application mode"] = APPLICATION_MODE[
            st.selectbox(
                "How you got admission",
                list(APPLICATION_MODE.keys()),
                help="Your admission route (JAMB, Direct Entry, Transfer, etc.)"
            )]

    user_input["Course"] = COURSE[
        st.selectbox("Course / Programme of study", list(COURSE.keys()))]

    # ------------------------------------------------------------------
    # 👨‍👩‍👧 SECTION 3 — FAMILY BACKGROUND
    # ------------------------------------------------------------------
    st.markdown("### 👨‍👩‍👧 3. Family Background")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Mother**")
        user_input["Mother's qualification"] = PARENT_QUALIFICATION[
            st.selectbox("Mother's highest education",
                         list(PARENT_QUALIFICATION.keys()), key="mq")]
        user_input["Mother's occupation"] = PARENT_OCCUPATION[
            st.selectbox("Mother's occupation",
                         list(PARENT_OCCUPATION.keys()), key="mo")]
    with c2:
        st.markdown("**Father**")
        user_input["Father's qualification"] = PARENT_QUALIFICATION[
            st.selectbox("Father's highest education",
                         list(PARENT_QUALIFICATION.keys()), key="fq")]
        user_input["Father's occupation"] = PARENT_OCCUPATION[
            st.selectbox("Father's occupation",
                         list(PARENT_OCCUPATION.keys()), key="fo")]

    # ------------------------------------------------------------------
    # 💰 SECTION 4 — FINANCIAL STATUS
    # ------------------------------------------------------------------
    st.markdown("### 💰 4. Financial Status")
    c1, c2, c3 = st.columns(3)
    with c1:
        user_input["Scholarship holder"] = YES_NO[
            st.radio("Are you on a scholarship or bursary?",
                     list(YES_NO.keys()), horizontal=True, key="sch")]
    with c2:
        user_input["Tuition fees up to date"] = YES_NO[
            st.radio("Are your school fees fully paid?",
                     list(YES_NO.keys()), horizontal=True, key="fee", index=1)]
    with c3:
        user_input["Debtor"] = YES_NO[
            st.radio("Do you currently owe the school money?",
                     list(YES_NO.keys()), horizontal=True, key="debt")]

    # ------------------------------------------------------------------
    # 📚 SECTION 5 — FIRST SEMESTER PERFORMANCE
    # ------------------------------------------------------------------
    st.markdown("### 📚 5. First Semester Academic Performance")
    st.caption("A **curricular unit** = one course/subject on your semester result.")
    c1, c2, c3 = st.columns(3)
    with c1:
        user_input["Curricular units 1st sem (enrolled)"] = st.number_input(
            "Courses registered", 0, 20, 6, key="e1",
            help="How many courses you signed up for in 1st semester.")
    with c2:
        user_input["Curricular units 1st sem (evaluations)"] = st.number_input(
            "Courses you sat exams for", 0, 40, 6, key="ev1")
    with c3:
        user_input["Curricular units 1st sem (approved)"] = st.number_input(
            "Courses passed", 0, 20, 6, key="a1",
            help="How many courses you got at least an 'E' or pass mark in.")

    c1, c2 = st.columns(2)
    with c1:
        pct1 = st.slider(
            "Average score (%) — 1st semester", 0, 100, 65, key="g1",
            help="Your average percentage score across all courses."
        )
        st.caption(f"Grade: **{nigerian_grade_label(pct1)}**")
        user_input["Curricular units 1st sem (grade)"] = percentage_to_portuguese(pct1)
    with c2:
        user_input["Curricular units 1st sem (without evaluations)"] = st.number_input(
            "Courses you missed completely (no score at all)",
            0, 20, 0, key="w1")

    # ------------------------------------------------------------------
    # 📚 SECTION 6 — SECOND SEMESTER PERFORMANCE
    # ------------------------------------------------------------------
    st.markdown("### 📚 6. Second Semester Academic Performance")
    c1, c2, c3 = st.columns(3)
    with c1:
        user_input["Curricular units 2nd sem (enrolled)"] = st.number_input(
            "Courses registered", 0, 20, 6, key="e2")
    with c2:
        user_input["Curricular units 2nd sem (evaluations)"] = st.number_input(
            "Courses you sat exams for", 0, 40, 6, key="ev2")
    with c3:
        user_input["Curricular units 2nd sem (approved)"] = st.number_input(
            "Courses passed", 0, 20, 6, key="a2")

    pct2 = st.slider("Average score (%) — 2nd semester", 0, 100, 65, key="g2")
    st.caption(f"Grade: **{nigerian_grade_label(pct2)}**")
    user_input["Curricular units 2nd sem (grade)"] = percentage_to_portuguese(pct2)

    # ------------------------------------------------------------------
    # 🌍 SECTION 7 — ECONOMIC CONTEXT
    # ------------------------------------------------------------------
    st.markdown("### 🌍 7. Economic Context")
    user_input["Inflation rate"] = st.number_input(
        "Inflation rate (%) during the academic year",
        -5.0, 50.0, 1.4, step=0.1,
        help=(
            "The national inflation rate for that year. Nigeria's recent "
            "inflation has been high (25–35%), but the model was trained on "
            "data with inflation between about -1% and 4%. Keeping a value "
            "in that range gives the most reliable prediction."
        )
    )

    # ------------------------------------------------------------------
    # 🔮 PREDICT BUTTON
    # ------------------------------------------------------------------
    st.markdown("---")
    if st.button("🔮 Predict Dropout Risk", type="primary", use_container_width=True):
        # make sure columns are in the exact training order
        X = pd.DataFrame([user_input])[features]
        X_scaled = scaler.transform(X)
        model = all_models[model_choice]
        proba = model.predict_proba(X_scaled)[0, 1]
        pred = int(proba >= 0.5)

        st.markdown("## Result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Model Used", model_choice)
        c2.metric("Dropout Probability", f"{proba*100:.1f}%")
        c3.metric("Prediction",
                  "⚠️ At Risk" if pred else "✅ Likely to Graduate")

        st.progress(float(proba))

        if proba >= 0.7:
            st.error(
                "**HIGH dropout risk.** Recommend immediate intervention: "
                "academic counselling, financial aid review, and a mentor."
            )
        elif proba >= 0.4:
            st.warning(
                "**MODERATE risk.** Monitor this student closely. "
                "Consider tutoring, study groups, or financial support."
            )
        else:
            st.success(
                "**LOW risk.** Student is on track to graduate. "
                "Keep up the good work!"
            )

# ============================================================
# TAB 2 — MODEL COMPARISON
# ============================================================
with tab2:
    st.subheader("Performance of the three models on the test set")
    st.dataframe(
        comparison.style.highlight_max(axis=0, color='#c7e9c0')
                        .format("{:.4f}"),
        use_container_width=True,
    )
    st.caption(f"✅ Best model by ROC-AUC: **{best_name}**")

    st.markdown("### 📈 Visualisations")
    for img, cap in [
        ("comparison_bars.png", "Metric comparison across the three models"),
        ("roc_curves.png", "ROC curves and AUC scores"),
        ("confusion_matrices.png", "Confusion matrices (actual vs predicted)"),
        ("mi_scores.png", "Top features by Mutual Information score"),
    ]:
        try:
            st.image(img, caption=cap)
        except Exception:
            st.info(f"Run `python train.py` first to generate **{img}**.")

    st.markdown("### 📝 Interpretation of the results")
    st.markdown(
        """
        - **Logistic Regression** — a linear, fully interpretable baseline.
          Each feature's effect can be read directly from its coefficient.
        - **Random Forest** — an ensemble of decision trees that captures
          non-linear interactions between features. Best overall accuracy
          and ROC-AUC in this project.
        - **Support Vector Machine (RBF kernel)** — a margin-based method
          that achieved the **highest precision** (fewest false alarms) but
          slightly lower recall.
        """
    )

# ============================================================
# TAB 3 — ABOUT
# ============================================================
with tab3:
    st.markdown(
        """
        ### About this project

        A machine learning system that predicts the likelihood that a
        university student will drop out, based on their demographic,
        family, financial, and academic-performance information.

        #### How the 24 features are categorised

        The model is a flat vector of 24 numbers, but for usability we
        group the inputs into **seven sections**:

        | # | Section | Example features |
        |---|---|---|
        | 1 | Personal Information | Gender, Age, Marital status, Relocation |
        | 2 | Educational Background | Previous qualification, Admission mode, Course |
        | 3 | Family Background | Parents' education and occupation |
        | 4 | Financial Status | Scholarship, Fees paid, Debtor |
        | 5 | 1st Semester Performance | Courses registered, passed, average score |
        | 6 | 2nd Semester Performance | Courses registered, passed, average score |
        | 7 | Economic Context | Inflation rate |

        #### Grade scale conversion

        The model was trained on the Portuguese **0–20** grade scale.
        The app lets you enter your score as a Nigerian **percentage (0–100)**
        and converts it internally:

        > Portuguese grade = Percentage ÷ 5

        For example: **75%** → 15.0 (A), **50%** → 10.0 (Pass), **40%** → 8.0 (Fail).

        #### Methodology

        1. Data cleaning (duplicates, outliers, ambiguous classes removed)
        2. Feature selection using Mutual Information (kept 24 of 34 features)
        3. Standard scaling of numeric features
        4. Training and 5-fold cross-validation of three models
        5. Deployment with Streamlit

        #### Honest limitations

        - Training data is from a **Portuguese** university, not a Nigerian one.
          The Nigerian labels in this app map to the closest Portuguese
          categories, but the model would need retraining on real Nigerian
          university data for production use.
        - The model uses **first-year** performance data, so it predicts dropout
          *after* the first academic year, not at admission time.
        - Nigerian inflation rates (25–35%) fall outside the training range
          (-1% to 4%), which reduces prediction reliability for very recent years.
        """
    )
