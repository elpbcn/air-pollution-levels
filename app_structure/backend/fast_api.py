from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class RequestModel(BaseModel):
    city: str
    year: int

@app.post("/get_index/")
def get_index(request: RequestModel):
    # This is a placeholder for actual logic
    index = len(request.city) + request.year % 100
    return {"index": index}

@app.get("/")
def root():
    return {"greeting": "Hello"}
