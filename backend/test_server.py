from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/test', methods=['GET'])
def test():
    return {'message': 'Flask server is working!'}

if __name__ == '__main__':
    print("Starting test Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5002)
