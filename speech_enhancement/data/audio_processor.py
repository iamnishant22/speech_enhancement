import numpy as np
import librosa
import soundfile as sf
from scipy import signal

class AudioProcessor:
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160, win_length=400):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
    def load_audio(self, file_path):
        """Load audio file and resample if necessary."""
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio
    
    def save_audio(self, audio, file_path):
        """Save audio to file."""
        sf.write(file_path, audio, self.sample_rate)
    
    def compute_stft(self, audio):
        """Compute Short-Time Fourier Transform."""
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length,
                           win_length=self.win_length, window='hann')
        return stft
    
    def compute_istft(self, stft):
        """Compute Inverse Short-Time Fourier Transform."""
        audio = librosa.istft(stft, hop_length=self.hop_length,
                            win_length=self.win_length, window='hann')
        return audio
    
    def compute_magnitude_phase(self, stft):
        """Separate magnitude and phase from STFT."""
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        return magnitude, phase
    
    def combine_magnitude_phase(self, magnitude, phase):
        """Combine magnitude and phase back to complex STFT."""
        return magnitude * np.exp(1j * phase)
    
    def normalize_audio(self, audio):
        """Normalize audio to [-1, 1] range."""
        return audio / np.max(np.abs(audio))
    
    def add_noise(self, clean_audio, noise_audio, snr_db):
        """Add noise to clean audio at specified SNR."""
        # Calculate noise power
        clean_power = np.mean(clean_audio ** 2)
        noise_power = np.mean(noise_audio ** 2)
        
        # Calculate scaling factor for noise
        snr_linear = 10 ** (snr_db / 10)
        scale = np.sqrt(clean_power / (noise_power * snr_linear))
        
        # Add scaled noise to clean audio
        noisy_audio = clean_audio + scale * noise_audio
        
        return self.normalize_audio(noisy_audio)
    
    def preprocess_audio(self, audio):
        """Preprocess audio for model input."""
        # Compute STFT
        stft = self.compute_stft(audio)
        
        # Get magnitude and phase
        magnitude, phase = self.compute_magnitude_phase(stft)
        
        # Log magnitude
        log_magnitude = np.log1p(magnitude)
        
        return log_magnitude, phase
    
    def postprocess_audio(self, enhanced_magnitude, phase):
        """Postprocess model output to get audio."""
        # Convert back from log scale
        magnitude = np.expm1(enhanced_magnitude)
        
        # Combine with phase
        stft = self.combine_magnitude_phase(magnitude, phase)
        
        # Compute ISTFT
        audio = self.compute_istft(stft)
        
        return self.normalize_audio(audio) 