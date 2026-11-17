@echo off
echo ===============================================================
echo Starting AI Resume Screening & Candidate Ranking System
echo ===============================================================
echo Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "Resume Screening Backend (FastAPI)" .venv\Scripts\python run_backend.py

echo Starting Streamlit Recruiter Dashboard on http://localhost:8501 ...
start "Resume Screening Dashboard (Streamlit)" .venv\Scripts\streamlit run frontend/app.py --server.port 8501

echo System launched! Dashboard will open in your browser.
