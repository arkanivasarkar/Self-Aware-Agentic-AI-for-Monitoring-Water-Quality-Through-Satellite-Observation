import pyaudio
import numpy as np
from transformers import pipeline, WhisperProcessor
import pyttsx3


# Load Whisper model with English language setting
model_name = "openai/whisper-medium"
processor = WhisperProcessor.from_pretrained(model_name)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

pipe = pipeline("automatic-speech-recognition", model=model_name, 
            generate_kwargs={"language": "en", "forced_decoder_ids": forced_decoder_ids})


# Recording settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper works best at 16kHz
CHUNK = 1024
RECORD_SECONDS = 7


tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Adjust speech speed


def speech_to_text():
    audio = pyaudio.PyAudio()

    # Open stream for recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

    print("Recording...")
    frames = []

    # Capture audio data
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Convert to NumPy array
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0

    transcription = pipe(audio_data)
    return transcription["text"]
  

def text_to_speech(text):
    tts_engine.say(text)
    tts_engine.runAndWait()  # This blocks execution until speaking is done

