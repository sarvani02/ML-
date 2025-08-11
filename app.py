import streamlit as st
import pandas as pd
import pickle

st.title("Student Recommendation Prediction - Random Forest")

# Load saved model & encoders
try:
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("label.pkl", "rb") as f:
        label = pickle.load(f)
    with open("onehot.pkl", "rb") as f:
        onehot = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found. Please train the model first.")
    st.stop()

# Input form
st.subheader("Enter Student Details")
overall_grade = st.selectbox("OverallGrade", onehot.categories_[0])
obedient = st.selectbox("Obedient", onehot.categories_[1])
research_score = st.number_input("ResearchScore", min_value=0.0)
project_score = st.number_input("ProjectScore", min_value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([[overall_grade, obedient, research_score, project_score]],
                            columns=["OverallGrade", "Obedient", "ResearchScore", "ProjectScore"])

    cat_transformed = onehot.transform(input_df[["OverallGrade", "Obedient"]])
    cat_df = pd.DataFrame(cat_transformed, columns=onehot.get_feature_names_out(["OverallGrade", "Obedient"]))

    final_df = pd.concat([input_df[["ResearchScore", "ProjectScore"]].reset_index(drop=True),
                          cat_df.reset_index(drop=True)], axis=1)

    final_scaled = scaler.transform(final_df)
    pred = model.predict(final_scaled)[0]
    pred_label = label.inverse_transform([pred])[0]
    st.success(f"Prediction: {pred_label}")

    # Confidence
    proba = model.predict_proba(final_scaled)[0]
    st.write(f"Confidence: {max(proba)*100:.2f}%")

    # Feature Importance 
    import matplotlib.pyplot as plt
    importances = model.feature_importances_
    features = final_df.columns
    plt.figure(figsize=(8, 4))
    plt.barh(features, importances)
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    st.pyplot(plt)
