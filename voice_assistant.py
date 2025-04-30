import streamlit as st
import sounddevice as sd
import soundfile as sf
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import pyttsx3
import os
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
token = os.getenv("HF_TOKEN")
login(token)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load model
processor = Wav2Vec2Processor.from_pretrained("benax-rw/KinyaWhisper")
model = Wav2Vec2ForCTC.from_pretrained("benax-rw/KinyaWhisper")

# TTS
tts = pyttsx3.init()
tts.setProperty('rate', 150)
tts.setProperty('volume', 1.0)

def speak(text):
    st.text(f"Assistant: {text}")
    tts.say(text)
    tts.runAndWait()

def record_audio(filename="input.wav", duration=5, fs=16000):
    st.text("🎙️ Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write(filename, audio, fs)
    st.success("✅ Recording complete!")

def transcribe_audio(filename="input.wav"):
    audio_input, _ = sf.read(filename)
    input_values = processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.decode(predicted_ids[0]).lower().strip()

# QA dictionary
qa_pairs = {
    "amakuru yawe": "Ni meza, urakoze!",
    "witwa nde": "Nitwa Umufasha wawe.",
    "wamfasha": "Yego, ndahari kugufasha!",
    "murakoze": "Murakaza neza!",
}

st.title("🗣️ Kinyarwanda Voice Assistant")

if st.button("🎤 Record and Transcribe"):
    record_audio()
    user_text = transcribe_audio()
    st.text(f"You said: {user_text}")
    if user_text in qa_pairs:
        speak(qa_pairs[user_text])
    else:
        speak("Ndasaba imbabazi, sinumva neza.")
