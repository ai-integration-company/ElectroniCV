import sqlite3

#user_id + file_id(tg-bot creates unique file_id's)
USER_DATA='user_data.db'

def init_db():
    conn = sqlite3.connect(USER_DATA)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS files (id INTEGER PRIMARY KEY, user_id TEXT, excel_file TEXT)''')
    conn.commit()
    conn.close()
    
def save_to_db(user_id, file_id):
    conn = sqlite3.connect(USER_DATA)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO files (user_id, excel_file) VALUES (?, ?)',
                                   (user_id, file_id))
    conn.commit()
    conn.close()