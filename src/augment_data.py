from preprocessing import clean_text
from transformers import pipeline
import random

def generate_synthetic_summary(prompt):
    generator = pipeline('text-generation', 'distilgpt2', device=0)
    output = generator(prompt, max_length=200, num_return_sequences=1)
    return clean_text(output[0]['generated_text'])  


if __name__ == "__main__":
    roles = ['Machine Learning Engineer', 'Data Scientist', 'AI Researcher']
    levels = ['Associate', 'Mid senior']
    skills = ['pytorch nlp bert embeddings', 'hugging face transformers scikit-learn faiss', 'ethical ai bias mitigation fairlearn']
    locs = ['San Francisco CA', 'New York NY', 'Remote']
    prompts = [f"{random.choice(roles)} job: {random.choice(levels)} level requiring {random.choice(skills)} for AI projects in {random.choice(locs)}." for _ in range(5000)]
    synthetic_summaries = [generate_synthetic_summary(p) for p in prompts]