from flask import *

app = Flask(__name__)

@app.route('/hello', methods=['POST'])
def hello_world():
    res = {'apple': 16.0, 'banana': 11.0, 'orange': 14.0}
    return jsonify({'status': 'success', 'res': res})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)