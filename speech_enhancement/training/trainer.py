import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from ..models.transformer.speech_enhancer import SpeechEnhancer
from ..data.audio_processor import AudioProcessor

class SpeechEnhancementDataset(torch.utils.data.Dataset):
    def __init__(self, clean_files, noise_files, audio_processor, snr_range=(-5, 20)):
        self.clean_files = clean_files
        self.noise_files = noise_files
        self.audio_processor = audio_processor
        self.snr_range = snr_range
        
    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx):
        # Load clean audio
        clean_audio = self.audio_processor.load_audio(self.clean_files[idx])
        
        # Randomly select noise file
        noise_idx = np.random.randint(0, len(self.noise_files))
        noise_audio = self.audio_processor.load_audio(self.noise_files[noise_idx])
        
        # Random SNR
        snr = np.random.uniform(*self.snr_range)
        
        # Add noise
        noisy_audio = self.audio_processor.add_noise(clean_audio, noise_audio, snr)
        
        # Preprocess
        noisy_mag, _ = self.audio_processor.preprocess_audio(noisy_audio)
        clean_mag, _ = self.audio_processor.preprocess_audio(clean_audio)
        
        return torch.FloatTensor(noisy_mag.T), torch.FloatTensor(clean_mag.T)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.audio_processor = AudioProcessor()
        
        # Initialize model
        self.model = SpeechEnhancer(
            input_dim=config['input_dim'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(self.device)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for noisy_mag, clean_mag in tqdm(train_loader, desc='Training'):
            noisy_mag = noisy_mag.to(self.device)
            clean_mag = clean_mag.to(self.device)
            
            # Forward pass
            enhanced_mag = self.model(noisy_mag)
            
            # Compute loss
            loss = self.criterion(enhanced_mag, clean_mag)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for noisy_mag, clean_mag in tqdm(val_loader, desc='Validation'):
                noisy_mag = noisy_mag.to(self.device)
                clean_mag = clean_mag.to(self.device)
                
                # Forward pass
                enhanced_mag = self.model(noisy_mag)
                
                # Compute loss
                loss = self.criterion(enhanced_mag, clean_mag)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_files, val_files, noise_files, num_epochs):
        # Create datasets
        train_dataset = SpeechEnhancementDataset(
            train_files, noise_files, self.audio_processor
        )
        val_dataset = SpeechEnhancementDataset(
            val_files, noise_files, self.audio_processor
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4
        )
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            
            # Train
            train_loss = self.train_epoch(train_loader)
            print(f'Train Loss: {train_loss:.4f}')
            
            # Validate
            val_loss = self.validate(val_loader)
            print(f'Validation Loss: {val_loss:.4f}')
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.config['save_dir'], 'best_model.pth')
                )
            
            print('-' * 50) 