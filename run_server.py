from fastapi import FastAPI
app = FastAPI()
import genmodel

model = GenModel("bigscience/bloom-1b7")

@app.get("/")
async def main_fuction():
    return {"response":"The server is working."}

@app.get("/")
async def test_bloom(question:str):
    model.run_prompt(str)
