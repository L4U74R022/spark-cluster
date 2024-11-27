from flask import Flask, jsonify, request, render_template
import numpy
from num6 import predict

app = Flask(__name__)

@app.get("/")
def main():
    return render_template("index.html")

@app.get('/hello')
def hello():
    return jsonify({'message': 'Hello World!'})

@app.post('/predict')
def model_predict():
    img = request.files.get('image')
    print(img)
    result = predict(img)
    # n = numpy.array(img)
    print(result)
    return jsonify({'predicted_number': result})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')