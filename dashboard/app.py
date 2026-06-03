import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="AI Addiction Detector",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI-Powered Digital Addiction Detection")
st.markdown("### Using Random Forest Model (81% Accuracy)")
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    model_path = r"C:\Users\suved_8hu\OneDrive\Desktop\DigitalAddictionProject\models\addiction_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# Sidebar
st.sidebar.header("📊 Navigation")
page = st.sidebar.radio("Go to", ["Predictor", "Model Info", "About"])

if page == "Predictor":
    st.header("🎯 Addiction Risk Predictor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📱 Enter Your Usage")
        screen_time = st.number_input("Screen Time (minutes)", 0, 1440, 300)
        app_switches = st.number_input("App Switches", 0, 200, 50)
        social_time = st.number_input("Social Media Time (minutes)", 0, 1440, 120)
    
    with col2:
        st.subheader("📊 Your Summary")
        st.metric("Screen Hours", f"{screen_time/60:.1f} hrs")
        st.metric("Social Media %", f"{(social_time/screen_time*100):.1f}%" if screen_time>0 else "0%")
        st.metric("App Switches/hr", f"{app_switches/(screen_time/60):.0f}" if screen_time>0 else "0")
    
    if st.button("🔮 Predict Addiction Risk", type="primary"):
        if model is not None:
            features = np.array([[screen_time, app_switches, social_time]])
            pred = model.predict(features)[0]
            prob = model.predict_proba(features)[0]
            
            st.markdown("---")
            col3, col4 = st.columns(2)
            
            with col3:
                if pred == 1:
                    st.error(f"🔴 HIGH RISK")
                    st.metric("Addiction Probability", f"{prob[1]:.1%}")
                else:
                    st.success(f"🟢 LOW RISK")
                    st.metric("Addiction Probability", f"{prob[1]:.1%}")
            
            with col4:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob[1]*100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 60], 'color': "yellow"},
                            {'range': [60, 100], 'color': "salmon"}
                        ]
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("💡 Recommendations")
            if pred == 1:
                if social_time > screen_time * 0.5:
                    st.write("• 📱 Reduce social media - it's your biggest risk factor")
                if screen_time > 480:
                    st.write("• ⏰ Try to keep screen time under 8 hours")
                st.write("• 🧘 Take regular breaks every hour")
            else:
                st.write("• 👍 You're doing well! Keep monitoring your usage")
        else:
            st.error("❌ Model not found!")

elif page == "Model Info":
    st.header("ℹ️ Model Information")
    
    st.markdown("""
    ### Random Forest Classifier
    
    **Training Data:** Kaggle Mental Health Dataset
    - 500 users
    - 382 addicted, 118 normal
    
    **Features Used:**
    - Daily screen time (minutes)
    - Number of app switches
    - Social media time (minutes)
    
    **Performance:**
    - Test Accuracy: **81%**
    - Cross-validation: 73.4%
    
    **Feature Importance:**
    - Social Media Time: 44% (most important)
    - Screen Time: 33%
    - App Switches: 23%
    """)
    
    # Feature importance chart
    importance_data = pd.DataFrame({
        'Feature': ['Social Media Time', 'Screen Time', 'App Switches'],
        'Importance': [0.44, 0.33, 0.23]
    })
    fig = px.bar(importance_data, x='Importance', y='Feature', 
                 orientation='h', title="What Predicts Addiction?")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.header("📱 About This Project")
    st.markdown("""
    ### AI-Based Digital Addiction Detection
    
    **How it works:**
    1. Kaggle dataset (500 users) trained Random Forest model
    2. Model learned patterns: social media time is key
    3. This dashboard predicts your addiction risk
    
    **Technologies:**
    - Python (pandas, scikit-learn)
    - Random Forest Classifier (81% accuracy)
    - Streamlit Dashboard
    - Android App for data collection
    
    **Final Year Project**
    B.Tech Information Technology
    """)

st.markdown("---")
st.caption("Made with ❤️ for Digital Wellness")