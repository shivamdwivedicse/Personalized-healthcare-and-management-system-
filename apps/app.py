import streamlit as st
import pandas as pd
import joblib

# Load all saved files 
model = joblib.load('best_model.pkl')
le = joblib.load('label_encoder.pkl')
symptom_cols = joblib.load('symptom_columns.pkl')

df_discription = pd.read_csv("discription.csv")
df_discription['Disease'] = df_discription['Disease'].str.strip()

df_precaution = pd.read_csv("Disease precaution.csv")
df_precaution['Disease'] = df_precaution['Disease'].str.strip()
df_precaution['Disease'] = df_precaution['Disease'].replace('Peptic ulcer diseae', 'Peptic ulcer disease')
df_precaution['Disease'] = df_precaution['Disease'].str.replace(r'\s+', ' ', regex=True)

df_medication = pd.read_csv("medications.csv")
df_medication['Disease'] = df_medication['Disease'].str.strip()

df_diet = pd.read_csv("diets.csv")
df_diet['Disease'] = df_diet['Disease'].str.strip()
df_diet['Disease'] = df_diet['Disease'].replace({
    'Osteoarthritis': 'Osteoarthristis',
    '(vertigo) Paroxysmal Positional Vertigo': '(vertigo) Paroymsal Positional Vertigo'
})

#  Helper functions
def predict_disease(user_symptoms):
    input_vector = pd.DataFrame([[1 if col in user_symptoms else 0 for col in symptom_cols]], columns=symptom_cols)
    return model.predict(input_vector)[0]

def get_description(disease):
    row = df_discription[df_discription['Disease'] == disease]
    return row.iloc[0]['Description'] if not row.empty else "No description available"

def get_precautions(disease):
    row = df_precaution[df_precaution['Disease'] == disease]
    return row.iloc[0, 1:].dropna().tolist() if not row.empty else ["No precaution data"]

def get_medication(disease):
    row = df_medication[df_medication['Disease'] == disease]
    return row.iloc[0]['Medication'] if not row.empty else "No medication data"

def get_diet(disease):
    row = df_diet[df_diet['Disease'] == disease]
    return row.iloc[0]['Diet'] if not row.empty else "No diet data"

#  Streamlit UI 
st.set_page_config(page_title="Healthcare Recommendation System", page_icon="🩺", layout="centered")

st.title("🩺 Personalized Healthcare & Medicine Recommendation System")
st.write("Select your symptoms below and get an AI-powered health recommendation.")

selected_symptoms = st.multiselect("Select your symptoms:", options=symptom_cols)

if st.button("🔍 Predict Disease"):
    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom.")
    else:
        disease = predict_disease(selected_symptoms)
        description = get_description(disease)
        precautions = get_precautions(disease)
        medication = get_medication(disease)
        diet = get_diet(disease)

        st.success(f"### Predicted Disease: {disease}")
        
        st.subheader("📋 Description")
        st.write(description)

        st.subheader("💊 Medication")
        st.write(medication)

        st.subheader("⚠️ Precautions")
        for i, p in enumerate(precautions, 1):
            st.write(f"{i}. {p}")

        st.subheader("🥗 Diet Recommendation")
        st.write(diet)

        st.warning("⚠️ Disclaimer: This tool is for educational purposes only. Please consult a certified doctor for medical advice.")