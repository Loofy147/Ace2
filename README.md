# ACE - Dynamic Context Repository (DCR)

This repository contains the implementation of the **Dynamic Context Repository (DCR)**, a functional multi-tier caching system for managing contextual data. This project is the first concrete deliverable from the aspirational Agentic Context Engineering (ACE) system, focusing on building real, testable components.

## Core Component: Dynamic Context Repository
The DCR is a multi-tier caching system with the following features:
- **L1 Hot Cache**: An in-memory, fast-access cache for the most recently used items (implemented with `OrderedDict` for LRU).
- **L2 Warm Cache**: An in-memory cache for less frequently accessed items (also `OrderedDict` with LRU).
- **L3 Cold Storage**: A persistent, file-based cache for items evicted from L2.
- **L4 Archive**: A persistent, read-only archive for long-term storage.
- **Automatic Tiering**: Contexts are automatically promoted to hotter tiers on access and evicted to colder tiers when caches are full.

## Project Structure
The project is organized into the following directories:
- `src/ace/core`: Contains the core implementation of the DCR.
- `src/ace/api`: Contains a FastAPI wrapper that provides API access to the system's components.
- `docs`: Contains high-level architectural and planning documents.
- `tests`: Contains unit tests for the DCR.
- `ROADMAP.md`: Outlines the development plan and future goals.

## Running the CLI Demo
The project includes a command-line interface (CLI) that demonstrates the DCR's caching, eviction, and promotion mechanisms in action.

### Prerequisites
- Python 3.7+
- All dependencies from `requirements.txt`

### Installation
```bash
pip install -r requirements.txt
```

### Usage
```bash
python -m src.ace.core.implementation
```

## Running the API
The system can also be run as a REST API. Currently, the API provides a pass-through to the underlying (and largely aspirational) ACE system components, but it is architected to expose the DCR and other functional components as they are developed.

### Running the API Server
```bash
uvicorn src.ace.api.main:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

### API Endpoints
- **POST /context/adapt**: A pass-through endpoint to the placeholder ACE `process` method. **Note:** This does not yet fully utilize the functional DCR.
