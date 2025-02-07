from flask import Flask, render_template, request, jsonify
import time
from rag import init_llm_model, generate_tex, get_model, get_tokenizer

app = Flask(__name__)

model_initialized = False
messages = []

@app.route('/')
def index():
    return render_template('index.html')

def process_message(message):
    answer, source_url = generate_text(model=get_model(), tokenizer=get_tokenizer(), query=message)
    messages.append(f"Antwort: {answer}")
    messages.append(f"Die generierte Antwort hat diese Quelle genutzt: {source_url}")

@app.route('/start_model', methods=['POST'])
def start_model():
    global model_initialized
    if not model_initialized:
        init_llm_model()
        model_initialized = True
        return jsonify({"status": "Model initialized"})
    return jsonify({"status": "Model already initialized"})

# Route for the frontend to send a message to the backend
@app.route('/send', methods=['POST'])
def send():
    data = request.json
    message = data.get('message', '')
    if message:
        messages.append(message)
        process_message(message)
    return jsonify({"messages": messages})

# Route for the frontend to get the messages list
@app.route('/messages', methods=['GET'])
def get_messages():
    return jsonify({"messages": messages})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006, debug=True, use_reloader=False)
