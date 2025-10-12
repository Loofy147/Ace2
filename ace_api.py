import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Import the ACESystem and necessary enums
from ace_implementation import ACESystem, SecurityLevel

app = FastAPI(
    title="ACE Architecture API",
    version="2.0.0",
    description="An API for the Agentic Context Engineering (ACE) system.",
)

# Instantiate the ACE System
ace_system = ACESystem()

class ProcessRequest(BaseModel):
    input: str
    domain: Optional[str] = "general"
    security_level: int = 0

@app.on_event("startup")
async def startup_event():
    # Although ACESystem is not async in its __init__,
    # we can prepare for future async initializations
    # For now, the instance is already created synchronously
    print("ACE System is ready.")

@app.on_event("shutdown")
async def shutdown_event():
    await ace_system.shutdown()

@app.get("/")
def read_root():
    """Root endpoint to check if the API is running."""
    return {"message": "ACE API is running"}

@app.post("/context/adapt")
async def process_context(request: ProcessRequest):
    """
    Process input through the Adversarial Context Adaptation Layer (ACAL).
    """
    try:
        # Map the integer security level to the SecurityLevel enum
        try:
            sec_level = SecurityLevel(request.security_level)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid security level")

        # Process the input using the ACE system
        result = await ace_system.process(request.input, sec_level)

        if result['status'] == 'error':
            raise HTTPException(status_code=500, detail=result['error'])

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
