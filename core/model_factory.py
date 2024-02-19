from core.spectrogram_model import SpectrogramModel
from core.spectrum_model import SpectrumModel


def get_model(use_spectrum=False, use_mps=False):
    return SpectrumModel(64 * 384, use_mps) if use_spectrum else SpectrogramModel(64 * 256, use_mps)
