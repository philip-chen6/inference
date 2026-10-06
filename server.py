from fastapi import Depends, FastAPI, Request
from typing import Annotated
from pydantic import BaseModel
from contextlib import asynccontextmanager
from threading import Lock
from engine import InferenceEngine

generation_lock = Lock()

class GenerateRequest(BaseModel): #pydantic helps with serialization and validation of request data to match schemas
    prompt: str

@asynccontextmanager # creates lifespan as a context manager
async def lifespan(app: FastAPI):
    # load model
    print("initializing engine")
    engine = InferenceEngine("Qwen/Qwen2.5-0.5B-Instruct")
    app.state.engine = engine #store engine in app state
    yield
    # clean up and release resources
    print("shutting down")
    del app.state.engine

#fast API automatically passes in the Request object since it detects the type
async def get_engine(request: Request): 
    return request.app.state.engine
app = FastAPI(lifespan=lifespan)

@app.post("/generate") #Depends takes a provider function that fast API expects, instead of an actual instance 
def generate(request: GenerateRequest, engine: Annotated[InferenceEngine, Depends(get_engine)]): 
    with generation_lock: # ensures that only one request can access the model at a time for sequential inference
        return engine.generate(request.prompt)