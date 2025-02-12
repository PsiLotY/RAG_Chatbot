"""This module contains a flask webserver to interact with the RAG model.
"""

import threading
import uuid
import gc
from datetime import datetime
from time import time
from flask import Flask, render_template, request, jsonify, session
from torch import cuda
from rag import RAG
from chroma_functions import ChromaDB

app = Flask(__name__)
app.secret_key = "secrete"

MESSAGES_PER_USER = {}
MODEL_INITIALIZED = False 
RAG_MODEL = None
PROCESSING_USERS = {}
LOG_FILE = "chat_log.txt"

chromadb = ChromaDB()

def initialize_model():
    """Initialize the RAG model in a background thread."""
    global RAG_MODEL, MODEL_INITIALIZED
    MODEL_INITIALIZED = False 
    if RAG_MODEL is not None:
        del RAG_MODEL
        gc.collect()
        cuda.empty_cache()

    RAG_MODEL = RAG(
        # model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        # tokenizer_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        model_name="deepseek-ai/DeepSeek-V2-Lite-Chat",
        tokenizer_name="deepseek-ai/DeepSeek-V2-Lite-Chat",
    )
    MODEL_INITIALIZED = True
    print("Model successfully initialized!")

threading.Thread(target=initialize_model, daemon=True).start()

@app.route("/")
def index() -> str:
    """Serve the index page and assign a unique session ID."""
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
    return render_template("index.html")

@app.route("/model_status", methods=["GET"])
def model_status():
    """Check if the model is initialized."""
    return jsonify({"initialized": MODEL_INITIALIZED})

def log_interaction(user_id: str, message: str, response: str, source: str, duration: float):
    """Log the each query and response to a file."""
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        file.write(f"Selected User: {session['user_option']}\n")
        file.write(f"User: {user_id}\n")
        file.write(f"Duration: {duration:.2f} seconds\n")
        distance = chromadb.get_last_distance()
        file.write(f"Document distance: {distance}\n")
        file.write(f"Query: {message}\n")
        file.write(f"Response: {response}\n")
        file.write(f"Source: {source}\n\n")
        file.write("-" * 50 + "\n\n")

@app.route("/submit_rating", methods=["POST"])
def submit_rating():
    """_summary_

    Returns:
        _type_: _description_
    """
    data = request.json
    rating = data.get("rating")
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "User ID not found in session"}), 400

    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(f"User {user_id} rated the response: {rating}/10\n")
        file.write("-" * 50 + "\n\n")

    return jsonify({"status": "success"})

@app.route("/set_option", methods=["POST"])
def set_option():
    """Set the user option in the session."""
    data = request.json
    option = data.get("option")
    session["user_option"] = option
    return jsonify({"status": "success"})

def process_message(user_id: str, message: str) -> None:
    """Process a user message and generate a response."""
    if PROCESSING_USERS.get(user_id, False):
        return

    PROCESSING_USERS[user_id] = True
    try:
        start_time = time()
        answer, source_url = RAG_MODEL.generate_text(query=message)
        end_time = time()
        duration = end_time - start_time

        # irrelevant because of send?
        if user_id not in MESSAGES_PER_USER:
            MESSAGES_PER_USER[user_id] = []

        MESSAGES_PER_USER[user_id].append(f"Antwort: {answer}")
        MESSAGES_PER_USER[user_id].append(f"Die generierte Antwort hat diese Quelle genutzt: {source_url}")
        log_interaction(user_id, message, answer, source_url, duration)

    except Exception as e:
        with open(LOG_FILE, "a", encoding="utf-8") as file:
            file.write(f"An error occurred: {e}\n")
            file.write("-" * 50 + "\n\n")

        MESSAGES_PER_USER[user_id].append("Leider ist ein Fehler aufgetreten. Das Model startet gerade neu. Bitte einmal F5 drücken oder die Seite neu laden.")
        initialize_model()
    finally:
        PROCESSING_USERS[user_id] = False

@app.route("/send", methods=["POST"])
def send():
    """Handle messages from the frontend."""
    if not MODEL_INITIALIZED:
        return jsonify({"error": "Model is still loading. Please wait."}), 503

    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "User ID not found in session"}), 400

    data = request.json
    message = data.get("message", "")
    if message:
        if user_id not in MESSAGES_PER_USER:
            MESSAGES_PER_USER[user_id] = []
        MESSAGES_PER_USER[user_id].append(message)
        process_message(user_id, message)
    return jsonify({"messages": MESSAGES_PER_USER[user_id]})

@app.route("/messages", methods=["GET"])
def get_messages():
    """Fetch messages for the current session user."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "User ID not found in session"}), 400

    return jsonify({"messages": MESSAGES_PER_USER.get(user_id, [])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6006, debug=True, use_reloader=False)
