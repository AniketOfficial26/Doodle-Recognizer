# Doodle Recognition API (Refactored Backend)

FastAPI backend for doodle recognition with optional Gemini (Google Generative AI) support.

## Quick Start

1) Create and activate a virtual environment (recommended)

```bash
python -m venv venv
# Windows PowerShell
./venv/Scripts/Activate.ps1
```

2) Install dependencies

```bash
pip install -r requirements.txt
# If requirements.txt is not present, minimally:
pip install fastapi uvicorn[standard] python-dotenv google-generativeai pillow requests
```

3) Configure environment variables

- Copy `.env.example` to `.env` and fill the values
- Never commit your real `.env`

```ini
GEMINI_API_KEY= # your Google Generative AI key
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
STABILITY_API_KEY= # optional
```

4) Run the server

```bash
python app.py
# runs on http://0.0.0.0:5001
```

## Project Structure

- `app.py` — FastAPI app factory and server entrypoint
- `routes.py` — API routes
- `services.py` — Gemini + Stability AI utilities and loaders
- `config.py` — configuration and CORS
- `preprocessing.py`, `models.py`, `schemas.py` — model & data helpers
- `.env.example` — template for environment variables

## Notes on Gemini

- `services.py` looks for `.env` in several places: next to `services.py`, project root, and `backend/.env`.
- Ensure `google-generativeai` is installed and `GEMINI_API_KEY` is set.

## Git Hygiene

- `.gitignore` excludes `.env`, virtual envs, caches, and large model files (e.g. `*.keras`).
- If you need to share the model, prefer a release asset or a download step instead of committing binaries.
