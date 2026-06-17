import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="HCV Progression Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

model = joblib.load("hcv_rf_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ==========================================================
# TITLE
# ==========================================================

st.title("🩺 HCV Progression Prediction System")

st.markdown("""
This system predicts the progression stage of Hepatitis C Virus (HCV)
using liver biomarker values and a Random Forest Machine Learning model.

### Disease Stages
- Blood Donor
- Suspect Blood Donor
- Hepatitis
- Fibrosis
- Cirrhosis
""")

st.divider()

# ==========================================================
# INPUT SECTION
# ==========================================================

st.header("Enter Patient Biomarker Values")

col1, col2 = st.columns(2)

with col1:

    ALB = st.number_input(
        "ALB (Albumin)",
        min_value=0.0,
        value=40.0
    )

    ALP = st.number_input(
        "ALP (Alkaline Phosphatase)",
        min_value=0.0,
        value=70.0
    )

    ALT = st.number_input(
        "ALT",
        min_value=0.0,
        value=30.0
    )

    AST = st.number_input(
        "AST",
        min_value=0.0,
        value=25.0
    )

    BIL = st.number_input(
        "BIL (Bilirubin)",
        min_value=0.0,
        value=10.0
    )

with col2:

    CHE = st.number_input(
        "CHE",
        min_value=0.0,
        value=8.0
    )

    CHOL = st.number_input(
        "CHOL",
        min_value=0.0,
        value=5.0
    )

    CREA = st.number_input(
        "CREA",
        min_value=0.0,
        value=80.0
    )
    PROT = st.number_input(
        "PROT",
        min_value=0.0,
        value=70.0
    )
    GGT = st.number_input(
        "GGT",
        min_value=0.0,
        value=30.0
    )

   
# ==========================================================
# PREDICTION BUTTON
# ==========================================================

if st.button("Predict HCV Stage"):

    # ======================================================
    # CREATE INPUT DATAFRAME
    # ======================================================

    input_data = pd.DataFrame(
        [[
            ALB,
            ALP,
            ALT,
            AST,
            BIL,
            CHE,
            CHOL,
            CREA,
            PROT,
            GGT
        ]],
        columns=[
            "ALB",
            "ALP",
            "ALT",
            "AST",
            "BIL",
            "CHE",
            "CHOL",
            "CREA",
            "PROT",
            "GGT"
        ]
    )

    # ======================================================
    # SCALING
    # ======================================================

    input_scaled = scaler.transform(input_data)

    # ======================================================
    # PREDICTION
    # ======================================================

    prediction = model.predict(input_scaled)

    predicted_class = label_encoder.inverse_transform(prediction)

    result = predicted_class[0]

    # ======================================================
    # PROBABILITY
    # ======================================================

    probabilities = model.predict_proba(input_scaled)[0]

    confidence = np.max(probabilities) * 100

    st.divider()

    # ======================================================
    # RESULT
    # ======================================================

    st.subheader("Prediction Result")

    st.success(f"Prediction Result: {result}")

    st.metric(
        "Confidence Score",
        f"{confidence:.2f}%"
    )

    # ======================================================
    # RECOMMENDED ACTION
    # ======================================================

    st.subheader("Recommended Action")

    if result == "Blood Donor":

        st.info(
            "The liver biomarker profile appears normal. "
            "Continue maintaining a healthy lifestyle and attend routine health screenings."
        )

    elif result == "Suspect Blood Donor":

        st.warning(
            "Some liver biomarker values may require further observation. "
            "It is recommended to repeat liver function tests and consult a healthcare professional if symptoms occur."
        )

    elif result == "Hepatitis":

        st.error(
            "The prediction suggests possible Hepatitis. "
            "Please consult a doctor or hepatologist for further medical evaluation and diagnostic testing."
        )

    elif result == "Fibrosis":

        st.error(
            "The prediction suggests possible liver fibrosis. "
            "Medical assessment is recommended to determine the extent of liver damage and appropriate treatment."
        )

    elif result == "Cirrhosis":

        st.error(
            "The prediction suggests possible liver cirrhosis. "
            "Seek immediate medical consultation for comprehensive evaluation and management."
        )

    # ======================================================
    # PROBABILITY CHART
    # ======================================================

    st.subheader("Prediction Probability Chart")

    class_names = label_encoder.classes_

    prob_df = pd.DataFrame({
        "Class": class_names,
        "Probability (%)": probabilities * 100
    })

    fig, ax = plt.subplots(figsize=(8, 4))

    bars = ax.bar(
        prob_df["Class"],
        prob_df["Probability (%)"]
    )

    ax.set_ylabel("Probability (%)")
    ax.set_xlabel("Disease Stage")
    ax.set_title("Prediction Probability Distribution")

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom"
        )

    plt.xticks(rotation=15)

    st.pyplot(fig)

    # ======================================================
    # PROBABILITY TABLE
    # ======================================================

    st.subheader("Probability Distribution Table")

    st.dataframe(
        prob_df.sort_values(
            by="Probability (%)",
            ascending=False
        ),
        use_container_width=True
    )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "Final Year Project: Prediction of HCV Progression Using Liver Biomarkers "
    "with Random Forest Machine Learning Model"
)
