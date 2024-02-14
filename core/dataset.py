import os
import random

import torch
import torchaudio
from pydub import AudioSegment
from torchaudio.transforms import MelSpectrogram
from torchaudio.transforms import Resample


def load_audio(file_path, audio_format='mp3'):
    if audio_format == 'mp3':
        audio = AudioSegment.from_mp3(file_path)
        samples = torch.tensor(audio.get_array_of_samples()).float()
        rate = audio.frame_rate
    else:
        samples, rate = torchaudio.load(file_path)

    if samples.dim() > 1 and samples.shape[0] == 2:
        samples = samples.mean(dim=0, keepdim=True)
    if samples.dim() == 1:
        samples = samples.unsqueeze(0)

    return samples, rate


def load_and_resample_audio(file_path, target_sample_rate=44100, audio_format='mp3'):
    samples, sample_rate = load_audio(file_path, audio_format)
    if sample_rate != target_sample_rate:
        resample_transform = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        samples = resample_transform(samples)
        sample_rate = target_sample_rate
    return samples, sample_rate


def mix_audio_samples(speech_waveform, bg_waveform, bg_volume=0.3):
    bg_waveform *= bg_volume

    if speech_waveform.shape[1] > bg_waveform.shape[1]:
        repeat_times = speech_waveform.shape[1] // bg_waveform.shape[1] + 1
        bg_waveform = bg_waveform.repeat(1, repeat_times)
    bg_waveform = bg_waveform[:, :speech_waveform.shape[1]]

    mixed_waveform = speech_waveform + bg_waveform

    return mixed_waveform


def audio_to_spectrogram(samples, sample_rate):
    spectrogram_transform = MelSpectrogram(sample_rate, n_mels=64, n_fft=2048)
    spectrogram = spectrogram_transform(samples.unsqueeze(0))
    return spectrogram


def process_folder(base_path, audio_format='mp3', limit_per_folder=100):
    spectrograms = []
    for root, dirs, files in os.walk(base_path):
        files_processed = 0
        for file_name in files:
            if file_name.endswith(f'.{audio_format}'):
                file_path = os.path.join(root, file_name)
                try:
                    samples, rate = load_audio(file_path, audio_format)
                    spectrogram = audio_to_spectrogram(samples, rate)
                    spectrograms.append(spectrogram)
                    files_processed += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            if files_processed >= limit_per_folder:
                break
    return spectrograms


def process_folder_with_bg(base_path, bg_path, limit_per_folder=100, base_audio_format='mp3', bg_audio_format='wav'):
    background_files = [os.path.join(root, name)
                        for root, dirs, files in os.walk(bg_path)
                        for name in files if name.endswith(bg_audio_format)]

    spectrograms = []
    for root, dirs, files in os.walk(base_path):
        files_processed = 0
        for file_name in files:
            if file_name.endswith(f'.{base_audio_format}'):
                file_path = os.path.join(root, file_name)
                try:
                    background_file_path = random.choice(background_files)
                    main_waveform, rate = load_and_resample_audio(file_path, 44100, base_audio_format)
                    bg_waveform, _ = load_and_resample_audio(background_file_path, rate, bg_audio_format)
                    mixed_waveform = mix_audio_samples(main_waveform, bg_waveform)
                    spectrogram = audio_to_spectrogram(mixed_waveform, rate)
                    spectrograms.append(spectrogram)
                    files_processed += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            if files_processed >= limit_per_folder:
                break
    return spectrograms
