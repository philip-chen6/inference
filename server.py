import asyncio
from fastapi import Depends, FastAPI, Request
from typing import Annotated
from pydantic import BaseModel
from contextlib import asynccontextmanager
from threading import Lock
from engine import InferenceEngine

generation_lock = Lock()

class GenerateRequest(BaseModel): #pydantic helps with serialization and validation of request data to match schemas
    prompt: str

async def worker(engine, queue):
    while True:
        # collect requests from queue, then make them (prompt, future) pair
        batch = [await queue.get()]

        # set a timeout (dynamic batching)
        try:
            async with asyncio.timeout(0.05):
                while len(batch) < 3:
                    batch.append(await queue.get())

        except TimeoutError:
            # continue below with however many requests are collected
            pass
        try:
            prompts = [prompt for prompt, future in batch]
            
            #to_thread so we can call synchronous blocking wfunctions outside of this thread to keep the asyncio event loop going
            results = await asyncio.to_thread(engine.batch_generate, prompts)
            for (prompt, future), output in zip(batch, results):
                if not future.done(): #so an already canceled request doesn't crash the worker.
                    future.set_result(output)
                # then replace future with the value and then do queue.task_done()
        except Exception as error:
            for prompt, future in batch:
                if not future.done():
                    future.set_exception(error)
        finally:
            for req in batch:
                queue.task_done()
    

@asynccontextmanager # creates lifespan as a context manager
async def lifespan(app: FastAPI):
    # load model
    print("initializing engine")
    engine = InferenceEngine("Qwen/Qwen2.5-0.5B-Instruct")
    app.state.engine = engine #store engine in app state
    queue = asyncio.Queue()
    app.state.queue = queue
    worker_task = asyncio.create_task(worker(engine, queue))
    
    try:
        yield
    finally:
        # clean up and release resources
        print("shutting down")
        # finish queued work, then ask the worker worker to stop and exit
        await queue.join()
        worker_task.cancel()
        try:
            await worker_task #wait for exit to actually happen. calling cancel only is a request
        except asyncio.CancelledError:
            pass
        del app.state.engine

#fast API automatically passes in the Request object since it detects the type
async def get_engine(request: Request): 
    return request.app.state.engine
app = FastAPI(lifespan=lifespan)

@app.post("/generate") #Depends takes a provider function that fast API expects, instead of an actual instance 
async def generate(body: GenerateRequest, request: Request): 
    queue = request.app.state.queue

    # get the event loop running in this thread and make a future associated with this request
    future = asyncio.get_running_loop().create_future()

    # add to queue
    await queue.put((body.prompt, future))
    # await the answer and return
    
    return await future