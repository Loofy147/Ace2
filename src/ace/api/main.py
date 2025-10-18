from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

import json
from typing import Dict
# Import the functional components
from ace.core.implementation import (
    DynamicContextRepository, Context, SecurityLevel,
    InputSanitizationEngine, PromptEngineeringLaboratory
)
from ace.llm.client import process_context_with_llm, format_prompt

app = FastAPI(
    title="ACE - Dynamic Context Repository API",
    version="0.1.0",
    description="An API for the functional components of the ACE system.",
)

# Instantiate components
dcr = DynamicContextRepository(base_storage_path="api_dcr_storage")
sanitization_engine = InputSanitizationEngine()
prompt_lab = PromptEngineeringLaboratory()

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
    ab_test_variant: Optional[str] = None

class CreateABTestRequest(BaseModel):
    test_name: str
    variants: Dict[str, str]

@app.on_event("startup")
def startup_event():
    print("ACE API is ready.")

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
def process_with_llm(context_id: str, ab_test_name: Optional[str] = None):
    """
    Retrieves a context and processes its content with an LLM.
    If ab_test_name is provided, a prompt variant from the test is used.
    """
    try:
        context = dcr.retrieve(context_id)
        if context is None:
            raise HTTPException(status_code=404, detail="Context not found")

        selected_variant = None

        if ab_test_name:
            variant_info = prompt_lab.get_prompt_variant(ab_test_name, context)
            if not variant_info:
                raise HTTPException(status_code=404, detail=f"A/B test '{ab_test_name}' not found.")

            prompt = variant_info["prompt"]
            selected_variant = variant_info["variant_name"]
        else:
            # Default behavior: format a standard prompt
            prompt = format_prompt(context)

        llm_response = process_context_with_llm(prompt)

        if llm_response.startswith("Error:"):
            raise HTTPException(status_code=500, detail=llm_response)

        return {
            "context_id": context_id,
            "llm_response": llm_response,
            "ab_test_variant": selected_variant
        }

    except Exception as e:
        # Catch exceptions from the LLM client, like missing API keys
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/prompts/ab_tests")
def create_ab_test(request: CreateABTestRequest):
    """
    Creates a new A/B test for prompts.
    """
    try:
        prompt_lab.create_ab_test(request.test_name, request.variants)
        return {"status": "success", "test_name": request.test_name, "variants": request.variants}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/prompts/ab_tests/{test_name}")
def get_ab_test_stats(test_name: str):
    """
    Retrieves the usage statistics for a given A/B test.
    """
    stats = prompt_lab.get_test_statistics(test_name)
    if stats is None:
        raise HTTPException(status_code=404, detail=f"A/B test '{test_name}' not found.")
    return {"test_name": test_name, "statistics": stats}
