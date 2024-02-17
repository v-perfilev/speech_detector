from core.audio_handler import AudioHandler
from core.dataset_handler import DatasetHandler
from core.file_handler import FileHandler
from core.spectrogram_mediator import SpectrogramMediator
from core.spectrogram_model import SpectrogramModel
from core.spectrum_mediator import SpectrumMediator
from core.spectrum_model import SpectrumModel


class DetectorFactory:
    is_spectrum_model = None

    def __init__(self, is_spectrum_model=False):
        self.is_spectrum_model = is_spectrum_model
        self.file_handler = FileHandler()
        self.dataset_handler = DatasetHandler()
        self.audio_handler = AudioHandler(target_spectrogram_shape=(64, 256),
                                          target_spectrum_size=1025,
                                          mix_background_volume=0.4)

    def get_audio_handler(self):
        return self.audio_handler

    def create_mediator(self):
        return SpectrumMediator(audio_handler=self.audio_handler,
                                file_handler=self.file_handler,
                                dataset_handler=self.dataset_handler) if self.is_spectrum_model \
            else SpectrogramMediator(audio_handler=self.audio_handler,
                                     file_handler=self.file_handler,
                                     dataset_handler=self.dataset_handler)

    def create_model(self):
        return SpectrumModel(64 * 384) if self.is_spectrum_model else SpectrogramModel(64 * 256)
