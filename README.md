# ACE - A/B Testing Framework

This repository contains the implementation of the **A/B Testing Framework**, a functional multi-tier caching system for managing contextual data, now with LLM integration and basic prompt injection detection. This project focuses on building real, testable components.

## Core Components
### 1. Dynamic Context Repository
A multi-tier caching system with L1/L2 in-memory LRU caches and L3/L4 file-based persistence.

### 2. LLM Integration Client
A client for processing contexts with OpenAI's API.

### 3. Input Sanitization Engine
A basic prompt injection detection system that checks for malicious keywords and command sequences.

### 4. Prompt Engineering Laboratory
A framework for A/B testing different prompt variations to evaluate their performance.

## Project Structure
- `src/ace/core`: Core implementation of the DCR, Sanitization Engine, and Prompt Lab.
- `src/ace/llm`: Client for communicating with the LLM API.
- `src/ace/api`: FastAPI wrapper for the system's functionality.
- `docs`: High-level architectural documents.
- `tests`: Unit tests for all functional components.
- `ROADMAP.md`: The development plan.

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
The API is available at `http://127.0.0.1:8000`.

### API Endpoints
- **`POST /context/store`**: Sanitizes and stores a new context.
- **`GET /context/retrieve/{context_id}`**: Retrieves a context.
- **`POST /prompts/ab_tests`**: Creates a new A/B test for prompts.
- **`POST /prompts/feedback`**: Records feedback for a response from an A/B test.
- **`GET /prompts/ab_tests/{test_name}`**: Retrieves aggregated usage and feedback statistics for an A/B test.
- **`POST /llm/process/{context_id}`**: Processes a context with an LLM. Using an `ab_test_name` will return a `trace_id` for feedback.

#### Example: Running an A/B Test with Feedback
1.  **Create an A/B test:**
    ```bash
    curl -X POST "http://127.0.0.1:8000/prompts/ab_tests" \
    -H "Content-Type: application/json" \
    -d '{
        "test_name": "summary_test",
        "variants": {
            "variant_a": "Summarize this: {context}",
            "variant_b": "Provide a brief summary of the following content: {context}"
        }
    }'
    ```

2.  **Store a context to test with:**
    ```bash
    curl -X POST "http://127.0.0.1:8000/context/store" \
    -H "Content-Type: application/json" \
    -d '{"content": {"text": "The quick brown fox jumps over the lazy dog."}}'
    ```
    *(This will return a `context_id`)*

3.  **Run the processing endpoint with the A/B test:**
    ```bash
    # This call will use a variant from the test and return a trace_id
    curl -X POST "http://127.0.0.1:8000/llm/process/{your_context_id_here}?ab_test_name=summary_test"
    ```
    *(The response will contain an `llm_response` and a `trace_id`)*

4.  **Record feedback for the response:**
    ```bash
    curl -X POST "http://127.0.0.1:8000/prompts/feedback" \
    -H "Content-Type: application/json" \
    -d '{"trace_id": "{your_trace_id_here}", "score": 0.9}'
    ```
    *(A score from 0.0 to 1.0 is expected)*

5.  **Check the aggregated results:**
    ```bash
    curl -X GET "http://127.0.0.1:8000/prompts/ab_tests/summary_test"
    ```
    *(This will return detailed stats including usage counts and average scores for each variant)*
