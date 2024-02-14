import os.path

from core.dataset import process_folder, process_folder_with_bg

speech_path = os.path.abspath('../datasets/speech')
sounds_path = os.path.abspath('../datasets/sounds')

speech_spectrograms = process_folder_with_bg(speech_path, sounds_path,
                                             base_audio_format='mp3', bg_audio_format='wav', limit_per_folder=25)
print(f"Processed {len(speech_spectrograms)} speech samples from each language folder.")

sounds_spectrograms = process_folder(sounds_path, audio_format='wav', limit_per_folder=10)
print(f"Processed {len(sounds_spectrograms)} environmental sound samples from each folder.")
