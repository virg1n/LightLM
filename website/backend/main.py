from flask import Flask, request, jsonify
from flask_cors import CORS

import torch

from model import enc, model_from_checkpoint, device

app = Flask(__name__)

CORS(app)

@app.route('/complete', methods=['POST'])
def complete():

    data = request.get_json()
    message = data.get('message')
    max_length = data.get('maxLength')

    gpt_return = enc.decode(model_from_checkpoint.generate(torch.tensor(enc.encode(message)).to(device).view(1, -1), int(max_length))[0].tolist())
    answer = gpt_return.split("<|endoftext|>")[0] 

    return jsonify({"response": answer[len(message)+1:]})

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5501)
