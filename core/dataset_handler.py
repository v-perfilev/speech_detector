import os
import random

import torch
import torch.nn.functional as F
import torchaudio
from pydub import AudioSegment
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchaudio.transforms import MelSpectrogram
from torchaudio.transforms import Resample


class DatasetHandler:
    speech_path = ""
    speech_audio_format = "mp3"
    speech_limit_per_folder = 100

    sounds_path = ""
    sounds_audio_format = "wav"
    sounds_limit_per_folder = 100

    target_spectrogram_shape = (64, 128)
    target_sample_rate = 44100
    batch_size = 64

    def set_speech_params(self, path, audio_format, limit_per_folder):
        self.speech_path = path
        self.speech_audio_format = audio_format
        self.speech_limit_per_folder = limit_per_folder

    def set_sounds_params(self, path, audio_format, limit_per_folder):
        self.sounds_path = path
        self.sounds_audio_format = audio_format
        self.sounds_limit_per_folder = limit_per_folder

    def set_target_spectrogram_shape(self, spectrogram_shape):
        self.target_spectrogram_shape = spectrogram_shape

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def read_datasets(self):
        speech_with_bg_spectrograms = self.__process_folder_with_bg(self.speech_path,
                                                                    self.sounds_path,
                                                                    self.speech_limit_per_folder,
                                                                    self.speech_audio_format,
                                                                    self.sounds_audio_format)

        print(f"Processed {len(speech_with_bg_spectrograms)} speech samples with background sounds.")

        speech_without_bg_spectrograms = self.__process_folder(self.speech_path,
                                                               self.speech_audio_format,
                                                               self.speech_limit_per_folder)
        print(f"Processed {len(speech_without_bg_spectrograms)} speech samples without background sounds.")

        sounds_spectrograms = self.__process_folder(self.sounds_path,
                                                    self.sounds_audio_format,
                                                    self.sounds_limit_per_folder)
        print(f"Processed {len(sounds_spectrograms)} environmental sound samples.")

        speech_spectrograms = speech_with_bg_spectrograms + speech_without_bg_spectrograms
        speech_spectrograms = self.__adjust_spectrogram_shape(speech_spectrograms)
        sounds_spectrograms = self.__adjust_spectrogram_shape(sounds_spectrograms)

        dataset = self.__convert_spectrograms_to_dataset(speech_spectrograms, sounds_spectrograms)
        return self.__split_dataset_to_data_loaders(dataset)

    def __load_audio(self, file_path, audio_format):
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

    def __load_and_resample_audio(self, file_path, audio_format):
        samples, sample_rate = self.__load_audio(file_path, audio_format)
        if sample_rate != self.target_sample_rate:
            resample_transform = Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            samples = resample_transform(samples)
            sample_rate = self.target_sample_rate
        return samples, sample_rate

    def __mix_audio_samples(self, speech_waveform, bg_waveform, bg_volume=0.3):
        bg_waveform *= bg_volume

        if speech_waveform.shape[1] > bg_waveform.shape[1]:
            repeat_times = speech_waveform.shape[1] // bg_waveform.shape[1] + 1
            bg_waveform = bg_waveform.repeat(1, repeat_times)
        bg_waveform = bg_waveform[:, :speech_waveform.shape[1]]

        mixed_waveform = speech_waveform + bg_waveform

        return mixed_waveform

    def __audio_to_spectrogram(self, samples, sample_rate):
        spectrogram_transform = MelSpectrogram(sample_rate, n_mels=64, n_fft=2048)
        spectrogram = spectrogram_transform(samples)
        return spectrogram

    def __process_folder(self, base_path, audio_format, limit_per_folder):
        spectrograms = []
        for root, dirs, files in os.walk(base_path):
            files_processed = 0
            for file_name in files:
                if file_name.endswith(f'.{audio_format}'):
                    file_path = os.path.join(root, file_name)
                    try:
                        samples, rate = self.__load_and_resample_audio(file_path, audio_format)
                        spectrogram = self.__audio_to_spectrogram(samples, rate)
                        spectrograms.append(spectrogram)
                        files_processed += 1
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                if files_processed >= limit_per_folder:
                    break
        return spectrograms

    def __process_folder_with_bg(self, base_path, bg_path, limit_per_folder, base_audio_format, bg_audio_format):
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
                        main_waveform, rate = self.__load_and_resample_audio(file_path, base_audio_format)
                        bg_waveform, _ = self.__load_and_resample_audio(background_file_path, bg_audio_format)
                        mixed_waveform = self.__mix_audio_samples(main_waveform, bg_waveform)
                        spectrogram = self.__audio_to_spectrogram(mixed_waveform, rate)
                        spectrograms.append(spectrogram)
                        files_processed += 1
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                if files_processed >= limit_per_folder:
                    break
        return spectrograms

    def __adjust_spectrogram_shape(self, spectrograms):
        adjusted_spectrograms = []

        for spectrogram in spectrograms:
            current_height, current_width = spectrogram.shape[1], spectrogram.shape[2]
            target_height, target_width = self.target_spectrogram_shape

            padding_height = max(0, target_height - current_height)
            padding_width = max(0, target_width - current_width)

            if padding_height > 0 or padding_width > 0:
                padding = [padding_width // 2, padding_width - padding_width // 2,
                           padding_height // 2, padding_height - padding_height // 2]
                spectrogram = F.pad(spectrogram, pad=padding, mode='constant', value=0)

            cropped_spectrogram = spectrogram[:, :target_height, :target_width]

            adjusted_spectrograms.append(cropped_spectrogram)

        return adjusted_spectrograms

    def __convert_spectrograms_to_dataset(self, speech_spectrograms, sounds_spectrograms):
        speech_labels = torch.ones(len(speech_spectrograms), dtype=torch.long)
        sounds_labels = torch.zeros(len(sounds_spectrograms), dtype=torch.long)
        all_labels = torch.cat((speech_labels, sounds_labels), dim=0)

        speech_spectrograms_tensor = torch.stack(speech_spectrograms)
        sounds_spectrograms_tensor = torch.stack(sounds_spectrograms)
        all_spectrograms_tensor = torch.cat((speech_spectrograms_tensor, sounds_spectrograms_tensor), dim=0)

        return TensorDataset(all_spectrograms_tensor, all_labels)

    def __split_dataset_to_data_loaders(self, dataset):
        dataset_size = len(dataset)
        train_size = int(dataset_size * 0.8)
        test_size = dataset_size - train_size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader
