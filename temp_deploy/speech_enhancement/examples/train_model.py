import os
import argparse
import numpy as np
from ..training.trainer import Trainer
from ..config import config

def main():
    parser = argparse.ArgumentParser(description='Train speech enhancement model')
    parser.add_argument('--clean_dir', type=str, required=True, help='Directory containing clean audio files')
    parser.add_argument('--noise_dir', type=str, required=True, help='Directory containing noise audio files')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    args = parser.parse_args()
    
    # Update config
    config['save_dir'] = args.output_dir
    config['num_epochs'] = args.epochs
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of audio files
    clean_files = [os.path.join(args.clean_dir, f) for f in os.listdir(args.clean_dir) 
                  if f.endswith(('.wav', '.mp3', '.flac'))]
    noise_files = [os.path.join(args.noise_dir, f) for f in os.listdir(args.noise_dir) 
                  if f.endswith(('.wav', '.mp3', '.flac'))]
    
    # Split into train and validation sets
    np.random.shuffle(clean_files)
    split_idx = int(0.8 * len(clean_files))
    train_files = clean_files[:split_idx]
    val_files = clean_files[split_idx:]
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Train model
    print("Starting training...")
    trainer.train(train_files, val_files, noise_files, config['num_epochs'])
    print("Training completed!")
    
    # Save final model
    torch.save(
        trainer.model.state_dict(),
        os.path.join(args.output_dir, 'final_model.pth')
    )
    print(f"Model saved to {os.path.join(args.output_dir, 'final_model.pth')}")

if __name__ == '__main__':
    main() 