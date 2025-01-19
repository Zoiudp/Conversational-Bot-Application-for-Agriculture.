import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from accelerate import Accelerator
import whisper
import pyaudio
import numpy as np
import wave
from gtts import gTTS
from IPython.display import Audio
import os
import re

RATE = 16000
CHANNELS = 1
CHUNK = 1024
DURATION = 5


def record_ambient_sound(rate=RATE, channels=CHANNELS, chunk=CHUNK, duration=DURATION):
    # Initialize PyAudio and open stream
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

    ambient_frames = []
    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        audio_data = np.frombuffer(data, dtype=np.int16)
        ambient_frames.append(audio_data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Calculate the mean amplitude of ambient sound
    ambient_audio = np.hstack(ambient_frames)
    ambient_mean = np.abs(ambient_audio).mean()

    return ambient_audio, ambient_mean

# Example usage
#ambient_audio, ambient_mean = record_ambient_sound()

# Função para gerar e reproduzir o áudio
def text_to_speech(text):
    # Cria o objeto gTTS
    tts = gTTS(text=text, lang='pt', slow=False)  # Ajuste o idioma com 'pt' para português, 'en' para inglês, etc.
    
    # Ensure the 'static' directory exists
    if not os.path.exists("model_output"):
        os.makedirs("model_output")

    # Salva o áudio em um arquivo temporário
    tts.save("model_output/output.mp3")

    print("Áudio gerado com sucesso!")
    
    return 


# Gera e toca o áudio
#text_to_speech(inference)

# Função para remover caracteres especiais
def remove_special_characters(text):
    # Remove caracteres especiais como '*', etc.
    return re.sub(r'[^\w\s.,!?]', '', text)  # Mantém apenas letras, números, espaços e alguns sinais de pontuação
