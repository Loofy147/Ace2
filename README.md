# ACE - Agentic Context Engineering System

This repository contains the implementation of the ACE (Agentic Context Engineering) system, a self-improving context management system with built-in adversarial robustness.

## Project Structure
The project is organized into the following directories:
- `src/ace/core`: Contains the core implementation of the ACE system.
- `src/ace/api`: Contains the FastAPI application for the REST API.
- `docs`: Contains documentation, including the ACE architecture.
- `tests`: Contains tests for the ACE system.

## ACE Architecture
For a detailed overview of the ACE architecture, please refer to the [docs/ace-architecture-v2.md](docs/ace-architecture-v2.md) document.

## Running the CLI
The ACE system can be run as a command-line interface (CLI) to demonstrate its capabilities.

### Prerequisites
- Python 3.7+
- `numpy`

### Installation
```bash
pip install -r requirements.txt
```

### Usage
```bash
python -m src.ace.core.implementation
```

## Running the API
The ACE system is also available as a REST API, providing a more accessible and versatile way to interact with the system.

### Prerequisites
- Python 3.7+
- `fastapi`
- `uvicorn`

### Installation
To install the required dependencies, run the following command:
```bash
pip install -r requirements.txt
```

### Running the API Server
To run the API server, use the following command:
```bash
uvicorn src.ace.api.main:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

### API Endpoints
- **GET /**: Returns a welcome message to confirm that the API is running.
- **POST /context/adapt**: Processes input through the Adversarial Context Adaptation Layer (ACAL).

#### Request Body for `/context/adapt`
- `input` (str): The input text to be processed.
- `domain` (str, optional): The domain of the input text (default: "general").
- `security_level` (int, optional): The security level of the input (default: 0).

#### Example Usage with `curl`
```bash
curl -X POST "http://127.0.0.1:8000/context/adapt" -H "Content-Type: application/json" -d '{
  "input": "Analyze the financial markets for Q4 2024",
  "domain": "finance",
  "security_level": 1
}'
```
