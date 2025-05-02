import numpy as np
import torch
import librosa
from ..models.transformer.speech_enhancer import SpeechEnhancementModel
from ..data.audio_processor import AudioProcessor

try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    print("Warning: PESQ package not available. Some metrics will be disabled.")

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False
    print("Warning: STOI package not available. Some metrics will be disabled.")

class Evaluator:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SpeechEnhancementModel(model_path)
        self.audio_processor = AudioProcessor()
    
    def compute_pesq(self, clean_audio, enhanced_audio):
        """Compute PESQ score if available."""
        if not PESQ_AVAILABLE:
            return None
        try:
            return pesq(16000, clean_audio, enhanced_audio, 'wb')
        except:
            return None
    
    def compute_stoi(self, clean_audio, enhanced_audio):
        """Compute STOI score if available."""
        if not STOI_AVAILABLE:
            return None
        try:
            return stoi(clean_audio, enhanced_audio, 16000, extended=False)
        except:
            return None
    
    def compute_snr(self, clean_audio, noisy_audio):
        """Compute Signal-to-Noise Ratio."""
        clean_power = np.mean(clean_audio ** 2)
        noise_power = np.mean((noisy_audio - clean_audio) ** 2)
        return 10 * np.log10(clean_power / noise_power)
    
    def compute_spectral_distance(self, clean_audio, enhanced_audio):
        """Compute spectral distance between clean and enhanced audio."""
        clean_stft = self.audio_processor.compute_stft(clean_audio)
        enhanced_stft = self.audio_processor.compute_stft(enhanced_audio)
        
        clean_mag = np.abs(clean_stft)
        enhanced_mag = np.abs(enhanced_stft)
        
        return np.mean(np.abs(clean_mag - enhanced_mag))
    
    def evaluate_file(self, clean_file, noisy_file):
        """Evaluate model performance on a single file."""
        # Load audio files
        clean_audio = self.audio_processor.load_audio(clean_file)
        noisy_audio = self.audio_processor.load_audio(noisy_file)
        
        # Preprocess
        noisy_mag, phase = self.audio_processor.preprocess_audio(noisy_audio)
        
        # Enhance
        enhanced_mag = self.model.enhance(noisy_mag)
        
        # Postprocess
        enhanced_audio = self.audio_processor.postprocess_audio(enhanced_mag, phase)
        
        # Compute metrics
        metrics = {
            'snr': self.compute_snr(clean_audio, enhanced_audio),
            'spectral_distance': self.compute_spectral_distance(clean_audio, enhanced_audio)
        }
        
        # Add optional metrics if available
        pesq_score = self.compute_pesq(clean_audio, enhanced_audio)
        if pesq_score is not None:
            metrics['pesq'] = pesq_score
            
        stoi_score = self.compute_stoi(clean_audio, enhanced_audio)
        if stoi_score is not None:
            metrics['stoi'] = stoi_score
        
        return metrics
    
    def evaluate_dataset(self, clean_files, noisy_files):
        """Evaluate model performance on a dataset."""
        results = {
            'snr': [],
            'spectral_distance': []
        }
        
        if PESQ_AVAILABLE:
            results['pesq'] = []
        if STOI_AVAILABLE:
            results['stoi'] = []
        
        for clean_file, noisy_file in zip(clean_files, noisy_files):
            metrics = self.evaluate_file(clean_file, noisy_file)
            for metric, value in metrics.items():
                results[metric].append(value)
        
        # Compute average metrics
        avg_results = {
            metric: np.mean(values)
            for metric, values in results.items()
        }
        
        return avg_results
    
    def generate_comparison(self, clean_file, noisy_file, output_dir):
        """Generate comparison audio files for listening tests."""
        # Load audio files
        clean_audio = self.audio_processor.load_audio(clean_file)
        noisy_audio = self.audio_processor.load_audio(noisy_file)
        
        # Preprocess
        noisy_mag, phase = self.audio_processor.preprocess_audio(noisy_audio)
        
        # Enhance
        enhanced_mag = self.model.enhance(noisy_mag)
        
        # Postprocess
        enhanced_audio = self.audio_processor.postprocess_audio(enhanced_mag, phase)
        
        # Save files
        self.audio_processor.save_audio(clean_audio, f'{output_dir}/clean.wav')
        self.audio_processor.save_audio(noisy_audio, f'{output_dir}/noisy.wav')
        self.audio_processor.save_audio(enhanced_audio, f'{output_dir}/enhanced.wav')
        
        return {
            'clean': f'{output_dir}/clean.wav',
            'noisy': f'{output_dir}/noisy.wav',
            'enhanced': f'{output_dir}/enhanced.wav'
        } 