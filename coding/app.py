import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import base64
import os

st.set_page_config(page_title="🚨 Fraud Risk Analyzer", layout="wide", page_icon="🚔")


def set_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: center;
        }}
        .title {{
            font-size: 3rem; color: #d7263d; text-align: center; font-weight: bold;
        }}
        .subtitle {{
            text-align: center; color: #333; font-size: 1.1rem; margin-bottom: 25px;
        }}
        .stDownloadButton>button {{
            background-color: #27ae60; color: white; border-radius: 8px;
            font-size: 1rem; padding: 0.5rem 1.2rem;
        }}
        </style>
    """, unsafe_allow_html=True)

bg_path = os.path.join(os.path.dirname(__file__), "bg.jpg")
set_bg_from_local(bg_path)


st.markdown('<div class="title">📊 Traffic Violation Detection and Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload traffic violation data to analyze fraud rates by location.</div>', unsafe_allow_html=True)
st.markdown("---")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "..", "models", "fraud_detection_model.pkl"))
le_violation_type = joblib.load(os.path.join(BASE_DIR, "..", "encoders", "le_violation_type.pkl"))
le_vehicle_type = joblib.load(os.path.join(BASE_DIR, "..", "encoders", "le_vehicle_type.pkl"))
le_fine_paid = joblib.load(os.path.join(BASE_DIR, "..", "encoders", "le_fine_paid.pkl"))

features = [
    'Violation_Type', 'Fine_Amount', 'Speed_Limit', 'Recorded_Speed',
    'Alcohol_Level', 'Driver_Age', 'Vehicle_Type', 'Previous_Violations', 'Fine_Paid'
]


uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    df['Fine_Paid'] = df['Fine_Paid'].astype(str).str.strip().replace({'no': 'No', 'yes': 'Yes'})
    df['Vehicle_Type'] = df['Vehicle_Type'].astype(str).str.strip().replace({'private': 'Private', 'commercial': 'Commercial'})
    df['Violation_Type'] = df['Violation_Type'].astype(str).str.strip().replace({'speeding': 'Speeding', 'signal jump': 'Signal Jump'})

    for col, le in zip(['Fine_Paid', 'Vehicle_Type', 'Violation_Type'],
                       [le_fine_paid, le_vehicle_type, le_violation_type]):
        valid = list(le.classes_)
        df[col] = df[col].apply(lambda x: x if x in valid else valid[0])
        df[col] = le.transform(df[col])

    for col in features:
        if col not in df.columns:
            df[col] = 0

    df = df.fillna(0)

   
    X_input = df[features]
    df['Is_Fraudulent'] = model.predict(X_input)
    df['Fraud_Probability'] = model.predict_proba(X_input)[:, 1]

    # Location-wise summary
    if 'Location' in df.columns:
        st.subheader("📍 Location-wise Fraud Risk Summary")
        summary = df.groupby('Location').agg(
            Total_Records=('Is_Fraudulent', 'count'),
            Fraud_Cases=('Is_Fraudulent', 'sum')
        ).reset_index()
        summary['Risk (%)'] = (summary['Fraud_Cases'] / summary['Total_Records'] * 100).round(2)

        def classify_risk(risk):
            if risk >= 70:
                return 'High'
            elif risk >= 40:
                return 'Medium'
            else:
                return 'Low'

        summary['Risk_Level'] = summary['Risk (%)'].apply(classify_risk)

        st.dataframe(summary.sort_values(by='Risk (%)', ascending=False), use_container_width=True)

        
        st.subheader("📊 Fraud Risk Percentage by Location")
        fig = px.bar(
            summary.sort_values(by='Risk (%)', ascending=False),
            x='Location', y='Risk (%)', color='Risk_Level',
            color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'},
            title="Fraud Risk per Location",
            height=500
        )
        fig.update_layout(xaxis_title="Location", yaxis_title="Fraud Risk (%)", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # Download
        st.download_button("⬇️ Download Summary", data=summary.to_csv(index=False).encode('utf-8'),
                           file_name="location_fraud_summary.csv", mime='text/csv')
    else:
        st.warning("⚠️ 'Location' column not found in the uploaded file.")
else:
    st.info("📌 Please upload a `.csv` file to start analyzing.")
