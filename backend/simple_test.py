from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    print("Starting simple Flask app...")
    try:
        app.run(debug=False, host='127.0.0.1', port=5003)
    except Exception as e:
        print(f"Error starting Flask: {e}")

