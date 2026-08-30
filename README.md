# 🚗 AutoIntel — Automotive Performance Intelligence Agent

A locally hosted **Agentic RAG system** that routes natural language 
queries across a ChromaDB knowledge base and live SQLite telemetry 
database using LangChain, Ollama, and dynamic LLM-generated SQL.

## Architecture
- **Agent Layer** — Intent-based routing (DOCS vs DATA)
- **RAG Layer** — ChromaDB vector search on automotive knowledge
- **Data Layer** — SQLite telemetry database (150+ sensor readings)
- **LLM Layer** — Ollama (Llama 3.2) running fully locally
- **UI Layer** — Streamlit web application

## Tech Stack
LangChain · ChromaDB · SQLite · Sentence Transformers · 
Ollama · Streamlit · Python 3.11

## Setup
```bash
pip install -r requirements.txt
python setup.py
streamlit run app.py
```

## Sample Queries
- DATA: "Which vehicle has the highest average BHP?"
- DATA: "What is the average lateral g-force for BMW M3?"
- DOCS: "How does a turbocharger reduce lag?"
- DOCS: "Why do EVs have instant torque from zero RPM?"

## Key Concepts Demonstrated
- Agentic AI · RAG · Prompt Engineering · 
  LLM-generated SQL · Token Tracking · Local LLM Deployment
