import streamlit as st

st.title("🏥 Hospital Readmission Risk Predictor")

st.write("Adjust patient inputs to estimate readmission risk.")

# Inputs
age = st.slider("Patient Age", 0, 100, 50)
medications = st.slider("Number of Medications", 0, 50, 10)

# Simple example logic
risk_score = (age * 0.01) + (medications * 0.02)

# Output
st.subheader("Predicted Readmission Risk")
st.write(f"{round(risk_score * 100, 2)}%")

# Risk level
if risk_score > 1:
    st.error("High Risk Patient")
elif risk_score > 0.6:
    st.warning("Moderate Risk")
else:
    st.success("Low Risk")
