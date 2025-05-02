import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import yaml

def test_installation():
    print("Testing package installations...")
    print("-" * 50)
    
    # Test PyTorch
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
    
    # Test torchaudio
    print("\nTorchaudio version:", torchaudio.__version__)
    
    # Test numpy
    print("\nNumPy version:", np.__version__)
    print("NumPy array test:", np.array([1, 2, 3]))
    
    # Test librosa
    print("\nLibrosa version:", librosa.__version__)
    try:
        # Test librosa loading
        y, sr = librosa.load(librosa.ex('trumpet'))
        print("Librosa audio loading test: Success")
    except Exception as e:
        print("Librosa audio loading test: Failed -", str(e))
    
    # Test soundfile
    print("\nSoundFile version:", sf.__version__)
    try:
        # Create a test audio file
        test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        sf.write('test.wav', test_audio, 44100)
        print("SoundFile write test: Success")
    except Exception as e:
        print("SoundFile write test: Failed -", str(e))
        print("Note: You may need to install Visual C++ Redistributable for Visual Studio 2015-2022")
    
    # Test matplotlib
    print("\nMatplotlib version:", matplotlib.__version__)
    try:
        plt.figure()
        plt.plot([1, 2, 3], [1, 2, 3])
        plt.close()
        print("Matplotlib plotting test: Success")
    except Exception as e:
        print("Matplotlib plotting test: Failed -", str(e))
    
    # Test pandas
    print("\nPandas version:", pd.__version__)
    try:
        df = pd.DataFrame({'test': [1, 2, 3]})
        print("Pandas DataFrame test: Success")
    except Exception as e:
        print("Pandas DataFrame test: Failed -", str(e))
    
    # Test tqdm
    print("\nTesting tqdm...")
    try:
        for _ in tqdm(range(3)):
            pass
        print("Tqdm test: Success")
    except Exception as e:
        print("Tqdm test: Failed -", str(e))
    
    # Test scikit-learn
    try:
        import sklearn
        print("\nScikit-learn version:", sklearn.__version__)
        print("Scikit-learn test: Success")
    except ImportError:
        print("\nScikit-learn not installed")
    
    # Test PyYAML
    print("\nPyYAML version:", yaml.__version__)
    try:
        yaml.dump({'test': 1})
        print("PyYAML test: Success")
    except Exception as e:
        print("PyYAML test: Failed -", str(e))
    
    print("\nAll tests completed!")
    print("-" * 50)

if __name__ == "__main__":
    test_installation() 