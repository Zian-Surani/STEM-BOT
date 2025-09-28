# STEM Bot Streamlit Backend (Hugging Face)

This is a Streamlit app acting as backend + demo UI.

## Features
- Subject-aware answering (Science, Math, Social Science)
- Modes: MCQ / Short / Long
- PDF parsing
- Image OCR
- Retrieval (FAISS + embeddings)
- Math verification with SymPy
- Hugging Face Inference API

## Quick Start

```bash
cd stem_streamlit_backend
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt

streamlit run streamlit_app.py
```

Open http://localhost:8501
