# ACE - Dynamic Context Repository (DCR)

This repository contains the implementation of the **Dynamic Context Repository (DCR)**, a functional multi-tier caching system for managing contextual data, now with LLM integration and basic prompt injection detection. This project focuses on building real, testable components.

## Core Components
### 1. Dynamic Context Repository
A multi-tier caching system with L1/L2 in-memory LRU caches and L3/L4 file-based persistence.

### 2. LLM Integration Client
A client for processing contexts with OpenAI's API.

### 3. Input Sanitization Engine
A basic prompt injection detection system that checks for malicious keywords and command sequences.

## Project Structure
- `src/ace/core`: Core implementation of the DCR and Sanitization Engine.
- `src/ace/llm`: Client for communicating with the LLM API.
- `src/ace/api`: FastAPI wrapper for the system's functionality.
- `docs`: High-level architectural documents.
- `tests`: Unit tests for all functional components.
- `ROADMAP.md`: The development plan.

## Running the CLI Demo
A CLI demo of the DCR's caching, eviction, and promotion mechanisms.

### Installation
```bash
pip install -r requirements.txt
```

### Usage
```bash
python -m src.ace.core.implementation
```

## Running the API
The system's components are exposed via a REST API using FastAPI.

### API Setup
Before running the API, you must set the `OPENAI_API_KEY` environment variable:
```bash
export OPENAI_API_KEY='your_openai_api_key_here'
```

### Running the API Server
```bash
uvicorn src.ace.api.main:app --reload
```
The API is available at `http://1227.0.0.1:8000`.

### API Endpoints
- **`POST /context/store`**: Sanitizes and stores a new context. Returns the `context_id` and a `sanitization_status` ('passed' or 'flagged').
- **`GET /context/retrieve/{context_id}`**: Retrieves a context from the DCR.
- **`POST /llm/process/{context_id}`**: Processes a stored context with the configured LLM.

#### Example: Storing and Processing a Context
1.  **Store a context:**
    ```bash
    curl -X POST "http://127.0.0.1:8000/context/store" \
    -H "Content-Type: application/json" \
    -d '{"content": {"topic": "Renewable Energy", "data": "Solar panel efficiency has increased by 20% in the last decade."}}'
    ```
    *(This will return a `context_id` and `sanitization_status`)*

2.  **Process the stored context with the LLM:**
    ```bash
    curl -X POST "http://127.0.0.1:8000/llm/process/{your_context_id_here}"
    ```
