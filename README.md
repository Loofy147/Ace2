# ACE - Dynamic Context Repository (DCR)

This repository contains the implementation of the **Dynamic Context Repository (DCR)**, a functional multi-tier caching system for managing contextual data, now with LLM integration capabilities. This project is the first concrete deliverable from the aspirational Agentic Context Engineering (ACE) system, focusing on building real, testable components.

## Core Components
### 1. Dynamic Context Repository
The DCR is a multi-tier caching system with the following features:
- **L1 Hot Cache**: An in-memory, fast-access cache for the most recently used items (implemented with `OrderedDict` for LRU).
- **L2 Warm Cache**: An in-memory cache for less frequently accessed items (also `OrderedDict` with LRU).
- **L3 Cold Storage**: A persistent, file-based cache for items evicted from L2.
- **L4 Archive**: A persistent, read-only archive for long-term storage.
- **Automatic Tiering**: Contexts are automatically promoted to hotter tiers on access and evicted to colder tiers when caches are full.

### 2. LLM Integration Client
The system includes a client for processing contexts with a Large Language Model (currently supporting OpenAI's API).

## Project Structure
- `src/ace/core`: Contains the core implementation of the DCR.
- `src/ace/llm`: Contains the client for communicating with the LLM API.
- `src/ace/api`: Contains a FastAPI wrapper that exposes the system's functionality.
- `docs`: Contains high-level architectural and planning documents.
- `tests`: Contains unit tests for the DCR and LLM client.
- `ROADMAP.md`: Outlines the development plan and future goals.

## Running the CLI Demo
The project includes a command-line interface (CLI) that demonstrates the DCR's caching, eviction, and promotion mechanisms.

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
The API will be available at `http://127.0.0.1:8000`.

### API Endpoints
- **`POST /context/store`**: Stores a new context in the DCR.
- **`GET /context/retrieve/{context_id}`**: Retrieves a context from the DCR.
- **`POST /llm/process/{context_id}`**: Retrieves a context from the DCR and processes its content using the configured LLM.

#### Example: Storing and Processing a Context
1.  **Store a context:**
    ```bash
    curl -X POST "http://127.0.0.1:8000/context/store" \
    -H "Content-Type: application/json" \
    -d '{"content": {"topic": "Renewable Energy", "data": "Solar panel efficiency has increased by 20% in the last decade."}}'
    ```
    *(This will return a `context_id`)*

2.  **Process the stored context with the LLM:**
    ```bash
    curl -X POST "http://127.0.0.1:8000/llm/process/{your_context_id_here}"
    ```
