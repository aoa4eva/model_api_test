from fastapi import FastAPI
app = FastAPI()

@app.get("/")
async def main_fuction():
    return {"response":"The server is working."}
