import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Healthcare Recommendation System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD DATA & MODEL
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load('best_model.pkl')
    le = joblib.load('label_encoder.pkl')
    symptom_cols = joblib.load('symptom_columns.pkl')
    return model, le, symptom_cols

@st.cache_data
def load_datasets():
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
    return df_discription, df_precaution, df_medication, df_diet

model, le, symptom_cols = load_artifacts()
df_discription, df_precaution, df_medication, df_diet = load_datasets()

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def predict_disease(user_symptoms):
    input_vector = pd.DataFrame([[1 if col in user_symptoms else 0 for col in symptom_cols]], columns=symptom_cols)
    proba = model.predict_proba(input_vector)[0]
    disease = model.classes_[proba.argmax()]
    confidence = proba.max() * 100
    return disease, confidence

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

def format_symptom_name(s):
    return s.replace('_', ' ').title()

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        color: #1E88E5;
        font-size: 2.3rem;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #666;
        font-size: 1.05rem;
    }
    .result-card {
        background-color: #f8f9fa;
        color: #1a1a1a !important;
        border-radius: 12px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1E88E5;
        font-size: 1.02rem;
        line-height: 1.6;
    }
    .result-card, .result-card * {
        color: #1a1a1a !important;
    }
    .disease-banner {
        background: linear-gradient(90deg, #1E88E5, #42A5F5);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .disease-banner h2 {
        margin: 0;
        font-size: 1.8rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🩺 About This Project")
    st.write(
        "An ML-powered system that predicts the most likely disease "
        "from your symptoms and provides medicine, precaution, and diet "
        "recommendations."
    )
    st.markdown("---")
    st.markdown("### ⚙️ Tech Stack")
    st.write("- Python, scikit-learn\n- Random Forest Classifier\n- Streamlit")
    st.markdown("---")
    st.markdown("### 🔄 How It Works")
    st.write(
        "1. Select your symptoms\n"
        "2. ML model predicts the disease\n"
        "3. Get description, medicine,\n   precautions & diet advice"
    )
    st.markdown("---")
    st.caption("⚠️ Educational project only. Not a substitute for professional medical advice.")

# ============================================================
# MAIN HEADER
# ============================================================
st.markdown("""
    <div class="main-header">
        <h1>🩺 Personalized Healthcare & Medicine Recommendation System</h1>
        <p>Select your symptoms below to get an AI-powered health assessment</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# SYMPTOM SELECTION
# ============================================================
col1, col2 = st.columns([3, 1])

with col1:
    display_names = [format_symptom_name(s) for s in symptom_cols]
    name_to_col = dict(zip(display_names, symptom_cols))

    selected_display = st.multiselect(
        "🔎 Search and select your symptoms:",
        options=sorted(display_names),
        placeholder="Type to search symptoms (e.g. Itching, Headache, Fever)..."
    )
    selected_symptoms = [name_to_col[s] for s in selected_display]

with col2:
    st.metric("Symptoms Selected", len(selected_symptoms))

predict_btn = st.button("🔍 Predict Disease", use_container_width=True, type="primary")

st.markdown("---")

# ============================================================
# PREDICTION & RESULTS
# ============================================================
if predict_btn:
    if len(selected_symptoms) == 0:
        st.warning("⚠️ Please select at least one symptom to continue.")
    else:
        with st.spinner("Analyzing symptoms..."):
            disease, confidence = predict_disease(selected_symptoms)
            description = get_description(disease)
            precautions = get_precautions(disease)
            medication = get_medication(disease)
            diet = get_diet(disease)

        st.markdown(f"""
            <div class="disease-banner">
                <p style="margin:0; font-size:0.9rem; opacity:0.9;">PREDICTED CONDITION</p>
                <h2>{disease}</h2>
                <p style="margin:0; font-size:0.9rem;">Model Confidence: {confidence:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["📋 Description", "💊 Medication", "⚠️ Precautions", "🥗 Diet"])

        with tab1:
            st.markdown(f'<div class="result-card">{description}</div>', unsafe_allow_html=True)

        with tab2:
            st.markdown(f'<div class="result-card">{medication}</div>', unsafe_allow_html=True)

        with tab3:
            precaution_html = "<br>".join([f"✔️ {p.capitalize()}" for p in precautions])
            st.markdown(f'<div class="result-card">{precaution_html}</div>', unsafe_allow_html=True)

        with tab4:
            st.markdown(f'<div class="result-card">{diet}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.info("⚠️ **Disclaimer:** This tool is built for educational purposes only. "
                "It is not a substitute for professional medical diagnosis. "
                "Please consult a certified doctor for medical advice.")

        with st.expander("📊 Selected Symptoms Summary"):
            st.write(", ".join(selected_display))