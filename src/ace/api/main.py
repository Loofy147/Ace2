from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

import json
# Import the functional components
from ace.core.implementation import DynamicContextRepository, Context, SecurityLevel, InputSanitizationEngine
from ace.llm.client import process_context_with_llm

app = FastAPI(
    title="ACE - Dynamic Context Repository API",
    version="0.1.0",
    description="An API for the functional Dynamic Context Repository (DCR) component.",
)

# Instantiate the DCR and Sanitization Engine
dcr = DynamicContextRepository(base_storage_path="api_dcr_storage")
sanitization_engine = InputSanitizationEngine()

class StoreRequest(BaseModel):
    content: dict
    domain: Optional[str] = "general"
    security_level: int = 0

class StoreResponse(BaseModel):
    status: str
    context_id: str
    sanitization_status: str

class ProcessResponse(BaseModel):
    context_id: str
    llm_response: str

@app.on_event("startup")
def startup_event():
    print("DCR API is ready.")

@app.post("/context/store", response_model=StoreResponse)
def store_context(request: StoreRequest):
    """
    Sanitizes and stores a new context in the Dynamic Context Repository.
    """
    try:
        # Sanitize the input content before creating the context object
        # We serialize the dict to a string to check for multi-line injections etc.
        content_str = json.dumps(request.content)
        sanitization_result = sanitization_engine.sanitize(content_str)

        sec_level = SecurityLevel(request.security_level)

        new_context = Context(
            content=request.content,
            domain=request.domain,
            security_level=sec_level
        )

        context_id = dcr.store(new_context)

        return {
            "status": "success",
            "context_id": context_id,
            "sanitization_status": sanitization_result['status']
        }

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid security level")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/context/retrieve/{context_id}")
def retrieve_context(context_id: str):
    """
    Retrieves a context from the DCR by its ID.
    """
    try:
        context = dcr.retrieve(context_id)
        if context is None:
            raise HTTPException(status_code=404, detail="Context not found")
        return context.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/llm/process/{context_id}", response_model=ProcessResponse)
def process_with_llm(context_id: str):
    """
    Retrieves a context and processes its content with an LLM.
    """
    try:
        context = dcr.retrieve(context_id)
        if context is None:
            raise HTTPException(status_code=404, detail="Context not found")

        llm_response = process_context_with_llm(context)

        if llm_response.startswith("Error:"):
            raise HTTPException(status_code=500, detail=llm_response)

        return {"context_id": context_id, "llm_response": llm_response}

    except Exception as e:
        # Catch exceptions from the LLM client, like missing API keys
        raise HTTPException(status_code=500, detail=str(e))
