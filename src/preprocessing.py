import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().strip()
    tokens = [word for word in text.split() if word not in STOPWORDS]
    return ' '.join(tokens)

def load_and_clean_jobs(postings_path, skills_path, summary_path, sample_size=100000, chunk_size=100000):
    postings = pd.concat(pd.read_csv(postings_path, chunksize=chunk_size), ignore_index=True)
    skills = pd.read_csv(skills_path)
    summaries = pd.read_csv(summary_path)
    df = postings.merge(skills, on='job_link', how='left').merge(summaries, on='job_link', how='left')
    df = df.dropna(subset=['job_summary', 'job_skills'])
    df['cleaned_summary'] = df['job_summary'].apply(clean_text)
    df['cleaned_skills'] = df['job_skills'].apply(lambda x: [s.strip().lower() for s in str(x).split(',')])
    df = df.sample(n=sample_size, random_state=42)
    return df[['job_link', 'job_title', 'company', 'job_location', 'job_level', 'job_type', 'cleaned_summary', 'cleaned_skills']]

