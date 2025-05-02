import argparse
import os
from ..models.transformer.speech_enhancer import SpeechEnhancementModel
from ..data.audio_processor import AudioProcessor

def main():
    parser = argparse.ArgumentParser(description='Enhance audio using Transformer-based model')
    parser.add_argument('--input', type=str, required=True, help='Path to input audio file')
    parser.add_argument('--output', type=str, required=True, help='Path to save enhanced audio')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    args = parser.parse_args()
    
    # Initialize model and audio processor
    model = SpeechEnhancementModel(args.model)
    audio_processor = AudioProcessor()
    
    # Load and process audio
    noisy_audio = audio_processor.load_audio(args.input)
    noisy_mag, phase = audio_processor.preprocess_audio(noisy_audio)
    
    # Enhance audio
    enhanced_mag = model.enhance(noisy_mag)
    enhanced_audio = audio_processor.postprocess_audio(enhanced_mag, phase)
    
    # Save enhanced audio
    audio_processor.save_audio(enhanced_audio, args.output)
    print(f'Enhanced audio saved to {args.output}')

if __name__ == '__main__':
    main() 