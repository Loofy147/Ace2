from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Import the functional components
from ace.core.implementation import DynamicContextRepository, Context, SecurityLevel

app = FastAPI(
    title="ACE - Dynamic Context Repository API",
    version="0.1.0",
    description="An API for the functional Dynamic Context Repository (DCR) component.",
)

# Instantiate the DCR
dcr = DynamicContextRepository(base_storage_path="api_dcr_storage")

class StoreRequest(BaseModel):
    content: dict
    domain: Optional[str] = "general"
    security_level: int = 0

class StoreResponse(BaseModel):
    status: str
    context_id: str

@app.on_event("startup")
def startup_event():
    print("DCR API is ready.")

@app.post("/context/store", response_model=StoreResponse)
def store_context(request: StoreRequest):
    """
    Stores a new context in the Dynamic Context Repository.
    """
    try:
        sec_level = SecurityLevel(request.security_level)

        new_context = Context(
            content=request.content,
            domain=request.domain,
            security_level=sec_level
        )

        context_id = dcr.store(new_context)

        return {"status": "success", "context_id": context_id}

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
