@echo off
echo Starting Tri-Tiered LLM Dashboard...
set HF_HUB_OFFLINE=1
.\venv\Scripts\streamlit.exe run ui/dashboard.py
pause
