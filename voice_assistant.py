from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio
import sounddevice as sd
import soundfile as sf
import time
import pyttsx3
from huggingface_hub import login
import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


token = os.getenv("HF_TOKEN")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
login(token)  # Replace with your Hugging Face token
# Load your model
processor = Wav2Vec2Processor.from_pretrained("benax-rw/KinyaWhisper")
model = Wav2Vec2ForCTC.from_pretrained("benax-rw/KinyaWhisper")
tts = pyttsx3.init()
tts.setProperty('rate', 150)  # Set speech rate
tts.setProperty('volume', 1.0)  # Set volume level (0.0 to 1.0)

def speak(text):
    print("Assistant:", text)
    tts.say(text)
    tts.runAndWait()

def transcribe_with_custom_model(audio_path):
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    # Mono
    input_values = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values

    # Predict
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    print("You said:", transcription)
    return transcription.lower().strip()

def record_audio(filename ="input.wav", duration=5,fs=16000):
    print("Listening...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, )
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    sf.write(filename, audio, fs)  # Save as WAV file

def transcribe_audio(filename ="input.wav"):
    #Load audio
    audio_input,_=sf.read(filename)
    input_values = processor(audio_input,return_tensors="pt",sampling_rate=16000).input_values
    #Get logits

    with torch.no_grad():
        logits = model(input_values).logits

    #decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    print("You said:", transcription)
    return transcription.lower().strip()

qa_pairs = {
    "amakuru yawe": "Ni meza, urakoze!",
    "witwa nde": "Nitwa Umufasha wawe.",
    "wamfasha": "Yego, ndahari kugufasha!",
    "murakoze": "Murakaza neza!",
}
while True:
    try:
        record_audio()
        user_text = transcribe_audio()
        print("You said:", user_text)
        if user_text in qa_pairs:
            speak(qa_pairs[user_text])
        else:
            speak("Ndasaba imbabazi, sinumva neza.")
    except KeyboardInterrupt:
        speak("Gusohoka. Murakoze!")
        break
    except Exception as e:
        speak(f"Habayeho ikosa: {e}")
        time.sleep(2)
