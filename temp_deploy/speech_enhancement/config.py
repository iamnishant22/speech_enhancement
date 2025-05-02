config = {
    # Model parameters
    'input_dim': 257,  # Number of frequency bins in STFT
    'd_model': 512,    # Dimension of the model
    'nhead': 8,        # Number of attention heads
    'num_layers': 6,   # Number of transformer layers
    'dropout': 0.1,    # Dropout rate
    
    # Training parameters
    'batch_size': 32,
    'learning_rate': 0.0001,
    'weight_decay': 0.0001,
    'num_epochs': 100,
    
    # Data parameters
    'sample_rate': 16000,
    'n_fft': 512,
    'hop_length': 160,
    'win_length': 400,
    
    # Paths
    'save_dir': 'checkpoints',
    'log_dir': 'logs',
    
    # Evaluation parameters
    'snr_range': (-5, 20),  # Range of SNR values for training
    'metrics': ['pesq', 'stoi', 'snr']
} 