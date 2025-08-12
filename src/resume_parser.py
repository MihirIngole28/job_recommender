import pdfplumber
from src.preprocessing import clean_text, load_and_clean_jobs

def parse_resume(file_path):
    if file_path.endswith('.pdf'):
        with pdfplumber.open(file_path) as pdf:
            text = ' '.join(page.extract_text() or '' for page in pdf.pages)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-16') as f:
            text = f.read()
    else:
        raise ValueError("Unsupported format")
    return clean_text(text)


if __name__ == "__main__":
    resume_files = ['data/resume1.txt', 'data/resume2.txt', 'data/resume3.txt', 'data/resume4.txt', 'data/resume5.txt']
    parsed_resumes = {file: parse_resume(file) for file in resume_files}
    print(parsed_resumes)
