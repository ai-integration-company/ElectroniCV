from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import db

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/get_data/{user_id}/{file_id}")
async def get_data(user_id:str, file_id: str):
    conn = db.get_db()
    data = db.get_file_data(conn, user_id, file_id)
    conn.close()
    if not data:
        raise HTTPException(status_code=404, detail="File not found.")
    
    return JSONResponse(content=[{
        "article": item.article,
        "name": item.name,
        "amount": item.amount,
        "price": item.price,
        "sum": item.sum
    } for item in data])
