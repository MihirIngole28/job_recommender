
# src/embedding_model.py
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity  # For baselines
import faiss
import numpy as np
import os

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  # M3 local; 'cuda' in Colab

def generate_embeddings(texts, model_name='all-MiniLM-L6-v2', batch_size=32):
    model = SentenceTransformer(model_name)
    model.to(device)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, device=device)
    return embeddings

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 for cosine (normalize if needed)
    faiss.normalize_L2(embeddings)  # For cosine similarity
    index.add(embeddings)
    return index


def recommend_jobs(resume_text, job_texts, job_df, top_k=10):
    resume_embedding = generate_embeddings([resume_text])
    index = faiss.read_index('data/processed/faiss_index.index')
    distances, indices = index.search(resume_embedding, top_k)
    recs = job_df.iloc[indices[0]].copy()
    recs['similarity_score'] = 1 - distances[0] / 2
    return recs[['job_title', 'company', 'job_location', 'job_level', 'similarity_score']]  # Added 'job_level'

    
# Example usage
if __name__ == "__main__":
    df = pd.read_csv('data/processed/cleaned_jobs_tech_aug.csv')
    job_texts = df['cleaned_summary'].tolist()  # Or combine with skills: df['cleaned_summary'] + ' ' + df['cleaned_skills'].str.join(' ')
    
    resume_text = "master computer science ml dl pytorch nlp bert"  # From parsed_resumes
    recs = recommend_jobs(resume_text, job_texts, df)
    print(recs)

#Avoid re-run
    '''
    job_embeddings = generate_embeddings(job_texts)
    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/job_embeddings.npy', job_embeddings)  # Save
    index = build_faiss_index(job_embeddings)
    faiss.write_index(index, 'data/processed/faiss_index.index')  # Save index  
    '''
