# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import IsolationForest
# import seaborn as sns
# import matplotlib.pyplot as plt

# # ---------------------------------------------------------
# # PAGE CONFIG
# # ---------------------------------------------------------
# st.set_page_config(
#     page_title="Insurance Claim Anomaly Detection",
#     layout="wide"
# )

# # ---------------------------------------------------------
# # ULTRA PREMIUM CSS (Animated Gradient + Glass Cards + Glow)
# # ---------------------------------------------------------
# st.markdown("""
# <style>

# @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

# * {
#     font-family: 'Poppins', sans-serif;
# }

# /* Animated Gradient Background */
# body {
#     background: linear-gradient(-45deg, #0F2027, #203A43, #2C5364, #0f0c29, #302b63, #24243e);
#     background-size: 600% 600%;
#     animation: gradientBG 20s ease infinite;
# }

# @keyframes gradientBG {
#     0% {background-position: 0% 50%;}
#     50% {background-position: 100% 50%;}
#     100% {background-position: 0% 50%;}
# }

# /* Glassmorphism Card */
# .card {
#     background: rgba(255, 255, 255, 0.07);
#     padding: 25px;
#     border-radius: 15px;
#     backdrop-filter: blur(10px);
#     box-shadow: 0px 8px 25px rgba(0,0,0,0.4);
#     border: 1px solid rgba(255,255,255,0.1);
#     transition: 0.25s;
# }

# .card:hover {
#     transform: translateY(-6px);
#     box-shadow: 0px 12px 35px rgba(0,0,0,0.6);
# }

# /* Premium Button */
# .stButton>button {
#     background: linear-gradient(135deg, #00c6ff, #0072ff);
#     color: white;
#     padding: 10px 25px;
#     border-radius: 10px;
#     border: none;
#     font-weight: 600;
#     transition: 0.3s;
# }

# .stButton>button:hover {
#     background: linear-gradient(135deg, #0072ff, #00c6ff);
#     transform: scale(1.05);
# }

# /* Glow Headings */
# h1, h2, h3 {
#     color: #ffffff !important;
#     text-shadow: 0px 0px 8px rgba(255,255,255,0.4);
# }

# /* Stylish File Uploader */
# .upload {
#     border: 2px dashed #6da8ff !important;
#     border-radius: 12px !important;
#     padding: 20px !important;
#     background: rgba(255,255,255,0.05);
# }

# </style>
# """, unsafe_allow_html=True)


# # ---------------------------------------------------------
# # MODEL LOAD
# # ---------------------------------------------------------
# @st.cache_resource
# def load_model():
#     df = pd.read_csv("insurance_claim_anomaly_50000.csv")

#     df_enc = pd.get_dummies(df, columns=["claim_type"], drop_first=False)

#     X = df_enc.drop("anomaly", axis=1)
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     model = IsolationForest(contamination=0.04, random_state=42)
#     model.fit(X_scaled)

#     return model, scaler, X.columns


# model, scaler, required_cols = load_model()


# # ---------------------------------------------------------
# # HEADER
# # ---------------------------------------------------------
# st.markdown("<h1 style='text-align:center;'> Insurance Claim Anomaly Detection Dashboard</h1>", unsafe_allow_html=True)
# st.markdown("<h4 style='text-align:center; color:#c7d5ff;'>Modern AI-powered fraud detection system for insurance claims.</h4>", unsafe_allow_html=True)
# st.markdown("<hr style='border:1px solid #ffffff22;'>", unsafe_allow_html=True)


# # ---------------------------------------------------------
# # LAYOUT
# # ---------------------------------------------------------
# col1, col2 = st.columns([1, 2])


# # =========================================================
# # LEFT PANEL — SINGLE CLAIM
# # =========================================================
# with col1:
#     st.markdown("<div class='card'>", unsafe_allow_html=True)
#     st.subheader(" Single Claim Evaluation")

#     claim_id = st.number_input("Claim ID", min_value=1, value=1001)
#     age = st.number_input("Customer Age", min_value=18, max_value=80, value=35)
#     tenure = st.number_input("Policy Tenure (years)", min_value=1.0, value=5.0)
#     amount = st.number_input("Claim Amount", min_value=500.0, value=20000.0)
#     duration = st.number_input("Claim Duration (days)", min_value=1.0, value=7.0)
#     claim_type = st.selectbox("Claim Type", ["Health", "Vehicle", "Property"])

#     ct_health = 1 if claim_type == "Health" else 0
#     ct_vehicle = 1 if claim_type == "Vehicle" else 0
#     ct_property = 1 if claim_type == "Property" else 0

#     row = np.array([
#         claim_id,
#         age,
#         tenure,
#         amount,
#         duration,
#         ct_health,
#         ct_vehicle,
#         ct_property
#     ]).reshape(1, -1)

#     scaled_row = scaler.transform(row)

#     if st.button("Evaluate Claim"):
#         pred = model.predict(scaled_row)
#         pred = 1 if pred == -1 else 0

#         if pred == 1:
#             st.error(" Fraud/Anomaly Detected!")
#         else:
#             st.success("✔ Normal Claim")

#     st.markdown("</div>", unsafe_allow_html=True)



# # =========================================================
# # RIGHT PANEL — CSV UPLOAD + VISUALIZATIONS
# # =========================================================
# with col2:
#     st.markdown("<div class='card'>", unsafe_allow_html=True)
#     st.subheader(" Batch Analysis (Upload CSV)")

#     uploaded = st.file_uploader("Upload Insurance Claims CSV", type=["csv"])

#     if uploaded:
#         df = pd.read_csv(uploaded)
#         st.write(" Uploaded File Preview:")
#         st.dataframe(df.head())

#         df_enc = pd.get_dummies(df, columns=["claim_type"], drop_first=False)

#         for col in required_cols:
#             if col not in df_enc.columns:
#                 df_enc[col] = 0

#         df_enc = df_enc[required_cols]

#         X_scaled = scaler.transform(df_enc)

#         df["prediction"] = model.predict(X_scaled)
#         df["prediction"] = df["prediction"].apply(lambda x: 1 if x == -1 else 0)

#         st.subheader(" Prediction Results")
#         st.dataframe(df)

#         # -----------------------------
#         # CHART: Normal vs Anomaly
#         # -----------------------------
#         st.subheader(" Normal vs Anomaly Distribution")
#         fig, ax = plt.subplots(figsize=(7, 4))
#         sns.countplot(x=df["prediction"], palette="coolwarm", ax=ax)
#         ax.set_xticklabels(["Normal", "Anomaly"])
#         st.pyplot(fig)

#         # -----------------------------
#         # CHART: Claim Amount
#         # -----------------------------
#         st.subheader(" Claim Amount Distribution")
#         fig2, ax2 = plt.subplots(figsize=(7, 4))
#         sns.histplot(df["claim_amount"], kde=True, color="#8ab4ff", ax=ax2)
#         st.pyplot(fig2)

#         csv_out = df.to_csv(index=False).encode("utf-8")
#         st.download_button("⬇ Download Output CSV", csv_out, "prediction_output.csv")

#     st.markdown("</div>", unsafe_allow_html=True)



import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Insurance Claim Anomaly Detection",
    layout="wide"
)

# ---------------------------------------------------------
# SUPER ULTRA PREMIUM CSS
# ---------------------------------------------------------


st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Perfect clean animated background */
body {
    background: linear-gradient(135deg, #0c0f1d, #1b2338, #283046);
    background-size: 300% 300%;
    animation: bgShift 18s ease infinite;
}

@keyframes bgShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Remove unwanted blank blocks */
.block-container {
    padding-top: 10px;
}

/* Clean Header */
.header-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
    color: #F1F3F7;
    letter-spacing: 0.5px;
    text-shadow: 0px 0px 12px rgba(255,255,255,0.25);
}

.header-sub {
    text-align: center;
    font-size: 18px;
    color: #cdd5e1;
    margin-top: -8px;
    margin-bottom: 20px;
}

/* Glassmorphism Cards */
.card {
    background: rgba(255,255,255,0.07);
    padding: 25px;
    border-radius: 16px;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0px 6px 22px rgba(0,0,0,0.45);
    transition: 0.3s ease;
}
.card:hover {
    transform: translateY(-4px);
    box-shadow: 0px 12px 35px rgba(0,0,0,0.6);
}

/* Section Title */
.section-title {
    font-size: 22px;
    font-weight: 600;
    color: #e8ecf3;
    margin-bottom: 10px;
    text-shadow: 0px 0px 8px rgba(255,255,255,0.18);
}

/* Button Styling */
.stButton > button {
    background: linear-gradient(135deg, #4ca1af, #2c3e50);
    border: none;
    padding: 10px 23px;
    font-weight: 600;
    color: white;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.35);
    transition: 0.2s ease;
}
.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(135deg, #2c3e50, #4ca1af);
}

/* Inputs Polished */
input, select {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #F1F3F7 !important;
}

</style>
""", unsafe_allow_html=True)



# ---------------------------------------------------------
# LOAD MODEL & TRAIN DATA
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("insurance_claim_anomaly_50000.csv")

    df_enc = pd.get_dummies(df, columns=["claim_type"], drop_first=False)
    X = df_enc.drop("anomaly", axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=0.04, random_state=42)
    model.fit(X_scaled)

    return model, scaler, X.columns


model, scaler, required_cols = load_model()


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("<h1 style='text-align:center;'> Insurance Claim Anomaly Detection Dashboard</h1>", unsafe_allow_html=True)
# st.markdown("<h4 style='text-align:center; color:#dce2ff;'>AI-powered fraud detection system with premium UI.</h4>", unsafe_allow_html=True)
# st.markdown("<hr>", unsafe_allow_html=True)




# ---------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------
col1, col2 = st.columns([1.1, 2])


# =========================================================
# LEFT SIDE – SINGLE CLAIM PREDICTION
# =========================================================
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔍 Single Claim Evaluation")

    claim_id = st.number_input("Claim ID", min_value=1, value=1001)
    age = st.number_input("Customer Age", min_value=18, max_value=80, value=35)
    tenure = st.number_input("Policy Tenure (years)", min_value=1.0, value=5.0)
    amount = st.number_input("Claim Amount", min_value=500.0, value=20000.0)
    duration = st.number_input("Claim Duration (days)", min_value=1.0, value=7.0)
    claim_type = st.selectbox("Claim Type", ["Health", "Vehicle", "Property"])

    ct_health = 1 if claim_type == "Health" else 0
    ct_vehicle = 1 if claim_type == "Vehicle" else 0
    ct_property = 1 if claim_type == "Property" else 0

    row = np.array([
        claim_id,
        age,
        tenure,
        amount,
        duration,
        ct_health,
        ct_vehicle,
        ct_property
    ]).reshape(1, -1)

    scaled_row = scaler.transform(row)

    if st.button("Evaluate Claim"):
        pred = model.predict(scaled_row)
        pred = 1 if pred == -1 else 0

        if pred == 1:
            st.error(" High Risk: Anomalous Claim Detected!")
        else:
            st.success("✔ Normal Claim Detected")

    st.markdown("</div>", unsafe_allow_html=True)



# =========================================================
# RIGHT SIDE — CSV UPLOAD + VISUALS
# =========================================================
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Batch Claim Analysis (Upload CSV)")

    uploaded = st.file_uploader("Upload Insurance Claims CSV", type=["csv"], key="upload")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(" Uploaded File Preview")
        st.dataframe(df.head())

        df_enc = pd.get_dummies(df, columns=["claim_type"], drop_first=False)

        for col in required_cols:
            if col not in df_enc.columns:
                df_enc[col] = 0

        df_enc = df_enc[required_cols]

        X_scaled = scaler.transform(df_enc)
        df["prediction"] = model.predict(X_scaled)
        df["prediction"] = df["prediction"].apply(lambda x: 1 if x == -1 else 0)

        st.subheader(" Prediction Results")
        st.dataframe(df)

        # NORMAL VS ANOMALY
        st.subheader(" Claims Distribution (Normal vs Anomaly)")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(x=df["prediction"], palette="coolwarm", ax=ax)
        ax.set_xticklabels(["Normal", "Anomaly"])
        st.pyplot(fig)

        # CLAIM AMOUNT CHART
        st.subheader(" Claim Amount Distribution")
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        sns.histplot(df["claim_amount"], kde=True, color="#7bb6ff", ax=ax2)
        st.pyplot(fig2)

        # DOWNLOAD
        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Results CSV", csv_out, "prediction_output.csv")

    st.markdown("</div>", unsafe_allow_html=True)

