# app/api.py (use FastAPI for API; install pip install fastapi uvicorn)
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/recommend")
async def api_recommend(resume_file: UploadFile = File(...), preferences: str = ""):
    resume_text = parse_resume(await resume_file.read())
    query_text = resume_text + " " + preferences
    recs = recommend_jobs(query_text, job_texts, df, top_k=10)
    return JSONResponse(recs.to_dict(orient='records'))

# Run: uvicorn app.api:app --reload