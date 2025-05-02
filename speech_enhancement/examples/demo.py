import os
import argparse
import numpy as np
from ..models.transformer.speech_enhancer import SpeechEnhancementModel
from ..data.audio_processor import AudioProcessor
from ..evaluation.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser(description='Demo of speech enhancement model')
    parser.add_argument('--clean_audio', type=str, required=True, help='Path to clean audio file')
    parser.add_argument('--noise_audio', type=str, required=True, help='Path to noise audio file')
    parser.add_argument('--snr', type=float, default=0, help='Signal-to-Noise Ratio in dB')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize components
    audio_processor = AudioProcessor()
    evaluator = Evaluator()
    
    # Load and process audio
    clean_audio = audio_processor.load_audio(args.clean_audio)
    noise_audio = audio_processor.load_audio(args.noise_audio)
    
    # Add noise to clean audio
    noisy_audio = audio_processor.add_noise(clean_audio, noise_audio, args.snr)
    
    # Save noisy audio
    audio_processor.save_audio(noisy_audio, os.path.join(args.output_dir, 'noisy.wav'))
    
    # Preprocess
    noisy_mag, phase = audio_processor.preprocess_audio(noisy_audio)
    
    # Enhance
    enhanced_mag = evaluator.model.enhance(noisy_mag)
    enhanced_audio = audio_processor.postprocess_audio(enhanced_mag, phase)
    
    # Save enhanced audio
    audio_processor.save_audio(enhanced_audio, os.path.join(args.output_dir, 'enhanced.wav'))
    
    # Evaluate
    metrics = evaluator.evaluate_file(args.clean_audio, os.path.join(args.output_dir, 'noisy.wav'))
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 50)
    print(f"SNR: {metrics['snr']:.2f} dB")
    print(f"Spectral Distance: {metrics['spectral_distance']:.4f}")
    
    if 'pesq' in metrics:
        print(f"PESQ: {metrics['pesq']:.2f}")
    if 'stoi' in metrics:
        print(f"STOI: {metrics['stoi']:.2f}")
    
    print("\nAudio files saved in:", args.output_dir)
    print("-" * 50)

if __name__ == '__main__':
    main() 