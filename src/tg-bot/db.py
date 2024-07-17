import sqlite3
from contextlib import closing
from pydantic import BaseModel
from typing import List

FILE_DATA = 'file_data.db'

column_mapping = {
    'article': 'Артикул',
    'name': 'Номенклатура',
    'amount': 'Количество',
    'price': 'Цена',
    'sum': 'Стоимость'
}


class FileData(BaseModel):

    user_id: str
    file_id: str
    article: str
    name: str
    amount: float
    price: float
    sum: float


class FileDataResponse(BaseModel):
    file_id: str
    data: List[FileData]


def init_db():
    conn = sqlite3.connect(FILE_DATA)
    cursor = conn.cursor()
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_data (
                user_id TEXT,
                file_id TEXT,
                article TEXT,
                name TEXT,
                amount REAL,
                price REAL,
                sum REAL
            )
            """)
    conn.commit
    conn.close

# def init_db():
#    with closing(sqlite3.connect(FILE_DATA)) as conn:
#        with conn:
#            conn.execute("""
#            CREATE TABLE IF NOT EXISTS file_data (
#                user_id TEXT,
#                file_id TEXT,
#                article TEXT,
#                name TEXT,
#                amount REAL,
#                price REAL,
#                sum REAL
#            )
#            """)
#


def get_db():
    conn = sqlite3.connect(FILE_DATA)
    conn.row_factory = sqlite3.Row
    return conn


def create_file_data(conn: sqlite3.Connection, user_id: str, file_id: str, data: List[dict]):
    with conn:
        for item in data:
            conn.execute("""
            INSERT INTO file_data (user_id, file_id, article, name, amount, price, sum) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, file_id, item['article'], item['name'], item['amount'], item['price'], item['sum']))


def get_file_data(conn: sqlite3.Connection, user_id: str, file_id: str) -> List[FileData]:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM file_data WHERE user_id = ? AND file_id = ?", (user_id, file_id))
    rows = cursor.fetchall()
    return [FileData(**row) for row in rows]
