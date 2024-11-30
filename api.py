from fastapi import FastAPI
from main import Vectorize_cosine_similarity
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
def Ready():
    return {"message":"ready"}

@app.get("/get_recommanded")
def run_get_recommanded(user_input):
    return JSONResponse (content=Vectorize_cosine_similarity(user_input))