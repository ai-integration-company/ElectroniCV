from flask import Flask, jsonify
import sqlite3
import pandas as pd

app = Flask(__name__)

@app.route('/data')
def data():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT excel_file FROM files ORDER BY id DESC LIMIT 1')
    file_path = cursor.fetchone()[0]
    conn.close()
    
    df = pd.read_excel(file_path)
    data = df.to_dict(orient='records')
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)
