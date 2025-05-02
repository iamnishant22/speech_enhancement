import numpy as np
import soundfile as sf
import librosa

def create_sample_audio(output_path='sample_noisy.wav', duration=10, sample_rate=16000):
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate speech-like signal (combination of sine waves to simulate speech)
    speech = (
        0.5 * np.sin(2 * np.pi * 440 * t) +  # Fundamental frequency (A4 note)
        0.3 * np.sin(2 * np.pi * 880 * t) +  # First harmonic
        0.2 * np.sin(2 * np.pi * 1320 * t) + # Second harmonic
        0.1 * np.sin(2 * np.pi * 1760 * t)   # Third harmonic
    )
    
    # Add some speech-like modulation
    speech *= (0.5 + 0.5 * np.sin(2 * np.pi * 5 * t))  # 5Hz modulation
    
    # Generate different types of noise
    # 1. White noise (constant across all frequencies)
    white_noise = np.random.normal(0, 0.1, len(speech))
    
    # 2. Pink noise (more energy in lower frequencies)
    pink_noise = np.random.normal(0, 0.05, len(speech)) * np.exp(-t/2)
    
    # 3. Brown noise (even more energy in lower frequencies)
    brown_noise = np.random.normal(0, 0.03, len(speech)) * np.exp(-t)
    
    # 4. Add some intermittent noise bursts
    noise_bursts = np.zeros_like(speech)
    burst_times = [2.0, 4.5, 7.0, 8.5]  # Times for noise bursts
    for burst_time in burst_times:
        burst_start = int(burst_time * sample_rate)
        burst_duration = int(0.3 * sample_rate)  # 300ms bursts
        noise_bursts[burst_start:burst_start + burst_duration] = np.random.normal(0, 0.4, burst_duration)
    
    # 5. Add some tonal noise (like a fan or machine)
    tonal_noise = 0.2 * np.sin(2 * np.pi * 120 * t)  # 120Hz tone
    
    # Combine all noise components
    total_noise = (
        white_noise + 
        pink_noise + 
        brown_noise + 
        noise_bursts + 
        tonal_noise
    )
    
    # Mix speech and noise
    noisy_signal = speech + total_noise
    
    # Normalize to prevent clipping
    noisy_signal = noisy_signal / np.max(np.abs(noisy_signal))
    
    # Save the noisy audio file
    sf.write(output_path, noisy_signal, sample_rate)
    
    print(f"Created sample file: {output_path}")
    print("\nFile contains:")
    print("1. Speech-like signal (440Hz tone with harmonics)")
    print("2. White noise (constant across frequencies)")
    print("3. Pink noise (more energy in lower frequencies)")
    print("4. Brown noise (even more energy in lower frequencies)")
    print("5. Intermittent noise bursts at 2.0s, 4.5s, 7.0s, and 8.5s")
    print("6. Tonal noise (120Hz constant tone)")
    print("\nTotal duration: 10 seconds")
    print("Sample rate: 16kHz")

if __name__ == "__main__":
    create_sample_audio() 