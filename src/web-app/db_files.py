import sqlite3
from contextlib import closing
from pydantic import BaseModel
from typing import List

FILE_DATA='file_data.db'


class FileData(BaseModel):
    file_id: str
    id: str
    name: str
    amount: float
    price: float
    sum: float

class FileDataResponse(BaseModel):
    file_id: str
    data: List[FileData]
    
    
def init_db():
    with closing(sqlite3.connect(FILE_DATA)) as conn:
        with conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS file_data (
                file_id TEXT,
                id TEXT,
                name TEXT,
                amount REAL,
                price REAL,
                sum REAL
            )
            """)

def get_db():
    conn = sqlite3.connect(FILE_DATA)
    conn.row_factory = sqlite3.Row
    return conn

def create_file_data(conn: sqlite3.Connection, file_id: str, data: List[dict]):
    with conn:
        for item in data:
            conn.execute("""
            INSERT INTO file_data (file_id, id, name, amount, price, sum) VALUES (?, ?, ?, ?, ?, ?)
            """, (file_id, item['id'], item['name'], item['amount'], item['price'], item['sum']))

def get_file_data(conn: sqlite3.Connection, file_id: str) -> List[FileData]:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM file_data WHERE file_id = ?", (file_id,))
    rows = cursor.fetchall()
    return [FileData(**row) for row in rows]