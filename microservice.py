from flask import Flask, request, send_file
import io
import pandas as pd

app = Flask(__name__)

@app.route('/compile', methods=['POST'])
def compile_png():
    if request.method == 'POST':
        file = request.files['file']
        # Process the PNG file (example)
        # Here, we create a dummy dataframe and save it as an excel file
        data = {'Column1': [1, 2, 3], 'Column2': [4, 5, 6]}
        df = pd.DataFrame(data)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        output.seek(0)
        return send_file(output, attachment_filename='output.xlsx', as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
