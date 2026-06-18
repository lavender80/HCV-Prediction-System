import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ======================================================
# PAGE CONFIGURATION
# ======================================================

st.set_page_config(
    page_title="HCV Progression Prediction",
    page_icon="🩺",
    layout="wide"
)

# ======================================================
# LOAD MODEL FILES
# ======================================================

model = joblib.load("xgboost_hcv_7biomarker.pkl")
scaler = joblib.load("scaler_7biomarker.pkl")
label_encoder = joblib.load("label_encoder_7biomarker.pkl")

# ======================================================
# HEADER
# ======================================================

st.title("🩺 HCV Progression Prediction System")

st.markdown("""
This system predicts the progression stage of Hepatitis C Virus (HCV)
using seven liver biomarkers and an XGBoost machine learning model.

**Biomarkers Used:**
- Albumin (ALB)
- Alkaline Phosphatase (ALP)
- Alanine Aminotransferase (ALT)
- Aspartate Aminotransferase (AST)
- Bilirubin (BIL)
- Cholinesterase (CHE)
- Gamma Glutamyl Transferase (GGT)
""")

st.divider()

# ======================================================
# INPUT SECTION
# ======================================================

st.subheader("Patient Liver Biomarker Information")

col1, col2 = st.columns(2)

with col1:

    ALB = st.number_input(
        "Albumin (ALB)",
        min_value=0.0,
        value=40.0,
        step=0.1
    )

    ALP = st.number_input(
        "Alkaline Phosphatase (ALP)",
        min_value=0.0,
        value=70.0,
        step=0.1
    )

    ALT = st.number_input(
        "Alanine Aminotransferase (ALT)",
        min_value=0.0,
        value=30.0,
        step=0.1
    )

    AST = st.number_input(
        "Aspartate Aminotransferase (AST)",
        min_value=0.0,
        value=30.0,
        step=0.1
    )

with col2:

    BIL = st.number_input(
        "Bilirubin (BIL)",
        min_value=0.0,
        value=10.0,
        step=0.1
    )

    CHE = st.number_input(
        "Cholinesterase (CHE)",
        min_value=0.0,
        value=8.0,
        step=0.1
    )

    GGT = st.number_input(
        "Gamma Glutamyl Transferase (GGT)",
        min_value=0.0,
        value=30.0,
        step=0.1
    )

# ======================================================
# PREDICTION BUTTON
# ======================================================

if st.button("Predict HCV Progression", use_container_width=True):

    # ==================================================
    # CREATE INPUT DATAFRAME
    # ==================================================

    input_data = pd.DataFrame(
        [[ALB, ALP, ALT, AST, BIL, CHE, GGT]],
        columns=[
            "ALB",
            "ALP",
            "ALT",
            "AST",
            "BIL",
            "CHE",
            "GGT"
        ]
    )

    # ==================================================
    # SCALING
    # ==================================================

    input_scaled = scaler.transform(input_data)

    # ==================================================
    # PREDICTION
    # ==================================================

    prediction = model.predict(input_scaled)

    probabilities = model.predict_proba(input_scaled)

    result = str(label_encoder.inverse_transform(prediction)[0]).strip()

    confidence = np.max(probabilities) * 100

    st.divider()

    # ==================================================
    # RESULT
    # ==================================================

    st.subheader("Prediction Result")

    st.success(f"Predicted Class: {result}")

    # ==================================================
    # CONFIDENCE SCORE
    # ==================================================

    st.subheader("Confidence Score")

    st.metric(
        label="Model Confidence",
        value=f"{confidence:.2f}%"
    )

    # ==================================================
    # PROGRESSION RISK
    # ==================================================

    st.subheader("Progression Risk")

    if "Blood Donor" in result and "Suspect" not in result:

        st.success("Low Risk")

    elif "Suspect Blood Donor" in result:

        st.warning("Low to Moderate Risk")

    elif "Hepatitis" in result:

        st.warning("Moderate Risk")

    elif "Fibrosis" in result:

        st.error("High Risk")

    elif "Cirrhosis" in result:

        st.error("Very High Risk")

    else:

        st.info("Risk level unavailable")

    # ==================================================
    # RECOMMENDED ACTION
    # ==================================================

    st.subheader("Recommended Action")

    result_clean = str(result).strip().lower()

    if "blood donor" in result_clean and "suspect" not in result_clean:

        st.info(
            "The liver biomarker profile appears normal. "
            "Continue maintaining a healthy lifestyle and attend routine health screenings."
        )

    elif "suspect blood donor" in result_clean:

        st.warning(
            "Some liver biomarker values may require further observation. "
            "It is recommended to repeat liver function tests and consult a healthcare professional if symptoms occur."
        )

    elif "hepatitis" in result_clean:

        st.error(
            "The prediction suggests possible Hepatitis. "
            "Please consult a doctor or hepatologist for further medical evaluation and diagnostic testing."
        )

    elif "fibrosis" in result_clean:

        st.error(
            "The prediction suggests possible liver fibrosis. "
            "Medical assessment is recommended to determine the extent of liver damage and appropriate treatment."
        )

    elif "cirrhosis" in result_clean:

        st.error(
            "The prediction suggests possible liver cirrhosis. "
            "Seek immediate medical consultation for comprehensive evaluation and management."
        )

    else:

        st.warning(f"No recommendation found for: {result}")

    # ==================================================
    # PROBABILITY DISTRIBUTION
    # ==================================================

    st.subheader("Prediction Probability Distribution")

    class_names = label_encoder.classes_

    prob_df = pd.DataFrame({
        "Class": class_names,
        "Probability (%)": probabilities[0] * 100
    })

    fig = px.bar(
        prob_df,
        x="Class",
        y="Probability (%)",
        text="Probability (%)",
        title="Class Prediction Probability"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==================================================
    # PROBABILITY TABLE
    # ==================================================

    st.subheader("Detailed Probability Table")

    st.dataframe(
        prob_df.style.format({
            "Probability (%)": "{:.2f}"
        }),
        use_container_width=True
    )

# ======================================================
# DISCLAIMER
# ======================================================

st.divider()

st.caption(
    "Disclaimer: This system is developed for academic and research purposes only. "
    "The prediction results should not be used as a substitute for professional medical diagnosis, treatment, or clinical decision-making."
)
