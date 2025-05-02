import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from speech_enhancement.data.audio_processor import AudioProcessor
from speech_enhancement.models.transformer.speech_enhancer import SpeechEnhancementModel

def generate_test_audio(duration=3, sample_rate=16000):
    """Generate a test audio signal with speech-like characteristics."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Generate a speech-like signal (combination of sine waves)
    f1, f2, f3 = 100, 200, 300  # Fundamental frequencies
    signal = (np.sin(2 * np.pi * f1 * t) + 
             0.5 * np.sin(2 * np.pi * f2 * t) + 
             0.25 * np.sin(2 * np.pi * f3 * t))
    
    # Add some amplitude modulation to make it more speech-like
    signal *= (1 + 0.5 * np.sin(2 * np.pi * 5 * t))
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    return signal

def generate_noise(duration=3, sample_rate=16000):
    """Generate white noise."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    noise = np.random.normal(0, 1, len(t))
    return noise / np.max(np.abs(noise))

def plot_spectrogram(audio, sample_rate, title):
    """Plot the spectrogram of an audio signal."""
    plt.figure(figsize=(10, 4))
    plt.specgram(audio, Fs=sample_rate, NFFT=512, noverlap=256)
    plt.colorbar(label='Intensity (dB)')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()

def main():
    # Initialize components
    audio_processor = AudioProcessor()
    model = SpeechEnhancementModel()
    
    # Generate test signals
    clean_audio = generate_test_audio()
    noise = generate_noise()
    
    # Add noise to clean audio
    noisy_audio = audio_processor.add_noise(clean_audio, noise, snr_db=0)
    
    # Save original signals
    sf.write('test_data/clean/clean.wav', clean_audio, 16000)
    sf.write('test_data/noise/noisy.wav', noisy_audio, 16000)
    
    # Process the noisy audio
    noisy_mag, phase = audio_processor.preprocess_audio(noisy_audio)
    enhanced_mag = model.enhance(noisy_mag)
    enhanced_audio = audio_processor.postprocess_audio(enhanced_mag, phase)
    
    # Save enhanced audio
    sf.write('test_data/output/enhanced.wav', enhanced_audio, 16000)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plot_spectrogram(clean_audio, 16000, 'Clean Audio')
    
    plt.subplot(3, 1, 2)
    plot_spectrogram(noisy_audio, 16000, 'Noisy Audio')
    
    plt.subplot(3, 1, 3)
    plot_spectrogram(enhanced_audio, 16000, 'Enhanced Audio')
    
    plt.tight_layout()
    plt.savefig('test_data/output/spectrograms.png')
    plt.close()
    
    print("Test completed! Check the 'test_data/output' directory for results.")

if __name__ == "__main__":
    main() 