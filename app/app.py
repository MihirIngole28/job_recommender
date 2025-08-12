# app/app.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.resume_parser import parse_resume
from src.embedding_model import recommend_jobs, generate_embeddings  # From Step 3
from src.advanced_features import bias_audit, explain_recs  # From Step 4
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import faiss
import numpy as np
from PIL import Image  # For multimodal if added

# Load precomputed data
df = pd.read_csv('data/processed/cleaned_jobs_tech_aug.csv')
job_texts = df['cleaned_summary'].tolist()
job_embeddings = np.load('data/processed/job_embeddings.npy')
index = faiss.read_index('data/processed/faiss_index.index')

st.title("AI-Powered Job Recommender")

# Input: Resume upload and preferences
resume_file = st.file_uploader("Upload Resume (PDF/TXT)", type=['pdf', 'txt'])
preferences = st.text_input("Job Preferences (e.g., machine learning engineer, San Francisco, entry-level)")

# Optional multimodal: CV image
cv_image = st.file_uploader("Optional: Upload CV Image for Professionalism Analysis", type=['jpg', 'png'])

if st.button("Get Recommendations"):
    if resume_file is not None:
        temp_path = f"temp_{resume_file.name}"
        with open(temp_path, "wb") as f:
            f.write(resume_file.getvalue())
        resume_text = parse_resume(temp_path)
        st.write("Parsed Resume Preview:", resume_text[:200] + "...")

        query_text = resume_text + " " + preferences

        # Generate recs (direct call; or use API and process response)
        recs = recommend_jobs(query_text, job_texts, df, top_k=10)  # Ensure assigned here

        # Now safe
        recs['recommended'] = 1
        bias_audit(recs)

        rec_features = np.column_stack((recs['similarity_score'], np.random.rand(len(recs), 2)))
        proxy_model = RandomForestRegressor().fit(rec_features, recs['recommended'])
        shap_values = explain_recs(proxy_model, rec_features)

        st.dataframe(recs[['job_title', 'company', 'job_location', 'similarity_score']])

        if cv_image:
            professionalism = analyze_cv_image(cv_image)
            st.write("CV Professionalism Score:", professionalism)

        os.remove(temp_path)
    else:
        st.error("Upload a resume to proceed.")