import signal
from io import BytesIO

import numpy as np
import pyaudio
import soundfile as sf
import torch

from core.audio_handler import AudioHandler
from core.model_factory import get_model

use_spectrum = False
threshold_sounds = False

audio_handler = AudioHandler()
model = get_model(use_spectrum=use_spectrum, use_mps=False)
model.load_model()


def list_microphones():
    p = pyaudio.PyAudio()
    print("List of available microphones:")
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev['maxInputChannels'] > 0:
            print(f"{i}: {dev['name']}")
    p.terminate()


def select_microphone():
    while True:
        idx = input("Enter the microphone index: ")
        if idx.isdigit() and 0 <= int(idx) < pyaudio.PyAudio().get_device_count():
            return int(idx)
        else:
            print("Invalid index. Please try again.")


def frames_to_audio(frames, rate):
    byte_data = b''.join(frames)
    audio_data = np.frombuffer(byte_data, dtype=np.int16)
    audio_file = BytesIO()
    sf.write(audio_file, audio_data, rate, format='wav')
    audio_file.seek(0)
    return audio_file


def predict_spectrum(samples):
    spectrum = audio_handler.audio_to_spectrum(samples)
    if threshold_sounds and spectrum.is_below_threshold():
        return None
    spectrum = spectrum.get_data()
    return model(spectrum.unsqueeze(0))


def predict_spectrogram(samples):
    spectrogram = audio_handler.audio_to_spectrogram(samples)
    if threshold_sounds and spectrogram.is_below_threshold():
        return None
    spectrogram = spectrogram.get_data()
    return model(spectrogram.unsqueeze(0))


def predict_audio(audio_file):
    samples, rate = audio_handler.load_audio(audio_file)
    if use_spectrum:
        prediction = predict_spectrum(samples)
    else:
        prediction = predict_spectrogram(samples)
    if prediction is None:
        return None
    predicted_class = torch.argmax(prediction, dim=1)
    return predicted_class


class AudioApp:
    analysis_window_seconds = 3
    shift_seconds = 1
    chunk = 1024
    audio_format = pyaudio.paInt16
    channels = 1
    rate = 44100
    running = True

    def __init__(self, microphone_idx):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.audio_format, channels=self.channels, rate=self.rate,
                                  input=True, frames_per_buffer=self.chunk,
                                  input_device_index=microphone_idx)
        self.frames = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def signal_handler(self, sig, frame):
        print("Stopping...")
        self.stop()

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()

    def listen(self):
        print("Recording...")
        while self.running:
            data = self.stream.read(self.chunk)
            self.frames.append(data)

            if len(self.frames) >= self.rate / self.chunk * self.analysis_window_seconds:
                audio_file = frames_to_audio(self.frames, self.rate)
                predicted_class = predict_audio(audio_file)
                match predicted_class:
                    case None:
                        print("Sound below threshold")
                    case 0:
                        print("Current sound: ENVIRONMENT")
                    case 1:
                        print("Current sound: SPEECH")
                self.frames = self.frames[int(self.rate / self.chunk * self.shift_seconds):]


if __name__ == "__main__":
    list_microphones()
    selected_microphone_idx = select_microphone()
    app = AudioApp(microphone_idx=selected_microphone_idx)
    signal.signal(signal.SIGINT, app.signal_handler)
    app.listen()
