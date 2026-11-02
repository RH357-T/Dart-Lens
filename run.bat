@echo off
cd /d "%~dp0"
if not exist .venv (
    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt
)
rem DART API 키를 여기 넣어두면 매번 입력하지 않아도 됩니다
rem set DART_API_KEY=여기에_키_입력
.venv\Scripts\python.exe -m streamlit run app.py
