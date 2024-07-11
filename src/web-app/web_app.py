from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import db_files

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

db_files.init_db()

#mapping from russian to english
column_mapping = {
    'Артикул': 'id',
    'Номенклатура': 'name',
    'Количество': 'amount',
    'Цена': 'price',
    'Стоимость': 'sum'
}

# Store data in-memory for simplicity. For production, change to a database.
file_data = {}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    #if file.content_type != 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
    #    raise HTTPException(status_code=400, detail="Invalid file type. Please upload an .xlsx file.")

    # Read the file into a pandas DataFrame
    try:
        df = pd.read_excel(file.file)
        # Rename the columns according to the mapping
        df.rename(columns=column_mapping, inplace=True)
        # Ensure all necessary columns are present
        for col in column_mapping.values():
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Missing required column: {col}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to read the .xlsx file.")
    
    # Set a unique file ID and store the database
    file_id = file.filename
    conn = db_files.get_db()
    db_files.create_file_data(conn, file_id, df.to_dict(orient='records'))
    conn.close()

    return {'file_id': file_id, 'data': df.to_dict(orient='records')}

@app.get("/get_data/{file_id}")
async def get_data(file_id: str):
    conn = db_files.get_db()
    data = db_files.get_file_data(conn, file_id)
    conn.close()
    if not data:
        raise HTTPException(status_code=404, detail="File not found.")
    
    return JSONResponse(content=[{
        "id": item.id,
        "name": item.name,
        "amount": item.amount,
        "price": item.price,
        "sum": item.sum
    } for item in data])
