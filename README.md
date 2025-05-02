# Speech Enhancement Using Transformer-Based Models

This project implements Transformer-based models for speech enhancement, focusing on improving speech clarity and reducing background noise. The implementation includes both traditional and Transformer-based approaches for performance comparison.

## Features

- Transformer-based speech enhancement models
- Real-time and near-real-time processing capabilities
- Support for various audio formats
- Comprehensive evaluation metrics
- Pre-trained models for quick deployment

## Project Structure

```
speech_enhancement/
├── data/                    # Data processing and loading utilities
├── models/                  # Model implementations
│   ├── transformer/        # Transformer-based models
│   └── traditional/        # Traditional enhancement methods
├── training/               # Training scripts and utilities
├── evaluation/             # Evaluation metrics and scripts
├── utils/                  # Utility functions
└── examples/               # Example usage scripts
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd speech-enhancement
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

```python
from training.trainer import Trainer

trainer = Trainer()
trainer.train()
```

### Inference

```python
from models.transformer import SpeechEnhancer

enhancer = SpeechEnhancer()
enhanced_audio = enhancer.enhance(audio_path)
```

## Performance Metrics

- Signal-to-Noise Ratio (SNR)
- Perceptual Evaluation of Speech Quality (PESQ)
- Short-Time Objective Intelligibility (STOI)
- Mean Opinion Score (MOS)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{speech_enhancement_transformer,
  title={Speech Enhancement Using Transformer-Based Models},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
``` 