from multiprocessing.pool import ThreadPool
import time
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from git import Tree
from openai import audio
from ray import get
import requests
from Inference import query_ollama_with_memory
from utils import text_to_speech
from BD_memory_utils import init_db, is_initialized
from accelerate import Accelerator
import os
import tempfile
import threading
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from AudioTranscriber import AudioTranscriber
import ssl

accelerator = Accelerator()
app = Flask(__name__)
CORS(app)

# Thread pool executor for managing threads
executor = ThreadPoolExecutor(max_workers=20)
# Global variable to track the listening state
is_listening = True

# Function to check initial listening state from frontend
# def check_initial_listening_state():
#     global is_listening
#     try:
#         # Make request to frontend (assuming it's running on port 3000)
#         response = requests.get('http://localhost:3000/api/listening-state')
#         if response.status_code == 200:
#             is_listening = response.json().get('isListening', True)
#     except:
#         # If frontend is not available, keep default value
#         pass

# # Check listening state when app starts
# check_initial_listening_state()

@app.route('/model_output/<filename>', methods=['GET'])
def get_audio(filename):
    try:
        audio_path = os.path.join(os.getcwd(), 'model_output', filename)
        return send_file(audio_path, as_attachment=True)
    except FileNotFoundError:
        return jsonify({'message': 'File not found'}), 404

@app.route('/audio_image', methods=['POST'])	
def process_data():
    print("Requisição recebida")

    if 'audio' not in request.files or 'image' not in request.files:
        return jsonify({'message': 'No audio or image file part in the request'}), 400

    audio_file = request.files['audio']
    image_file = request.files['image']

    if audio_file.filename == '' or image_file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    # Ensure the 'uploads' directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Define the paths to save the audio and image files
    audio_path = os.path.join('uploads', 'audio_file.wav')
    image_path = os.path.join('uploads', 'image_file.png')

    # Save the files to the static directory
    if audio_file and image_file:
        audio_file.save(audio_path)
        image_file.save(image_path)
    else:
        return jsonify({'message': 'Failed to upload audio or image file'}), 400

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_max_memory_allocated()
    
    start_time = time.time()

    # Initialize the database
    if not is_initialized():
        init_db()

    global is_listening
    if not is_listening:
        return jsonify({'message': 'Listening is disabled, no audio will be fetched.'}), 400

    try:
        # Transcriber initialization
        transcriber = AudioTranscriber(accelerator)

        # Handle audio transcription using ThreadPool
        with ThreadPool(processes=1) as pool:
            transcription_future = pool.apply_async(transcriber.transcribe_audio, (audio_path,))
            transcription = transcription_future.get()  # Wait for transcription

        # Clean CUDA resources
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_max_memory_allocated()

        print('Entrando em query_ollama_with_memory')
        with ThreadPool(processes=1) as pool:
            inference_future = pool.apply_async(query_ollama_with_memory, (transcription,))
            inference_response = inference_future.get()

        # Clean CUDA resources
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_max_memory_allocated()

        print('Entrando em Text to Speech')
        with ThreadPool(processes=1) as pool:
            pool.apply_async(text_to_speech, (inference_response,))
            pool.close()
            pool.join()
        
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_max_memory_allocated()

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Execution time calculation
    end_time = time.time()
    exec_time = end_time - start_time

    return jsonify({
        'message': inference_response,
        'audio_source': "http://127.0.0.1:5000/model_output/output.mp3",
        'tempo de execução': exec_time
    }), 200


@app.route('/set_listening_state', methods=['POST'])
def set_listening_state():
    global is_listening
    try:
        data = request.get_json()
        is_listening = data.get('isListening', True)
        return jsonify({'message': f'Listening state set to {is_listening}'}), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 400

if __name__ == '__main__':
    # Ensure you have the SSL certificate and key files
    cert_file = r'backend\cert.pem'
    key_file = r'backend\key.pem'
    app.run(host='0.0.0.0', port=5000, debug=True, ssl_context=(cert_file, key_file), threaded=True)