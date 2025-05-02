from flask import Flask, render_template, request, send_file, jsonify, Response
import os
from werkzeug.utils import secure_filename
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.ndimage import median_filter
import torch
import torchaudio
import json
import matplotlib.pyplot as plt
import io
import base64

# Try to import optional dependencies
try:
    import webrtcvad
    import collections
    import struct
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False
    print("WebRTC VAD not available. Voice activity detection will be disabled.")

try:
    import pesq
    import pystoi
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    print("PESQ and STOI metrics not available. Quality metrics will be disabled.")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'flac', 'm4a'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generate audio frames for VAD processing"""
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    while offset + n < len(audio):
        yield audio[offset:offset + n]
        offset += n

def is_speech(frame, vad, sample_rate):
    """Check if a frame contains speech using WebRTC VAD"""
    try:
        return vad.is_speech(frame.tobytes(), sample_rate)
    except:
        return True

def estimate_noise_floor(magnitude, n_frames=20):
    """Improved noise floor estimation using multiple techniques"""
    # Find frames with lowest energy
    frame_energies = np.mean(magnitude, axis=0)
    noise_frames = np.argsort(frame_energies)[:n_frames]
    
    # Estimate noise floor using minimum statistics
    noise_estimate = np.min(magnitude[:, noise_frames], axis=1, keepdims=True)
    
    # Apply smoothing to noise estimate
    noise_estimate = median_filter(noise_estimate, size=5)
    
    # Add adaptive noise tracking
    alpha = 0.95  # Smoothing factor
    noise_estimate = alpha * noise_estimate + (1 - alpha) * np.min(magnitude, axis=1, keepdims=True)
    
    return noise_estimate

def spectral_subtraction(audio, sr=16000, n_fft=2048, hop_length=512, mode='moderate'):
    """Advanced spectral subtraction with multiple noise reduction techniques"""
    # Initialize WebRTC VAD if available
    vad = None
    if WEBRTCVAD_AVAILABLE:
        vad = webrtcvad.Vad(3)  # Aggressiveness mode 3
    
    # Set parameters based on enhancement mode
    if mode == 'aggressive':
        alpha_low = 3.0
        alpha_high = 2.0
        beta = 0.05
        threshold_factor = 1.2
    elif mode == 'light':
        alpha_low = 1.5
        alpha_high = 1.2
        beta = 0.2
        threshold_factor = 1.8
    else:  # moderate (default)
        alpha_low = 2.0
        alpha_high = 1.5
        beta = 0.1
        threshold_factor = 1.5
    
    # Compute STFT with larger window for better frequency resolution
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # 1. Initial noise estimation
    noise_estimate = estimate_noise_floor(magnitude)
    
    # 2. Apply multi-band processing
    freq_bins = magnitude.shape[0]
    bands = [
        (0, freq_bins // 4),      # Low frequencies
        (freq_bins // 4, freq_bins // 2),  # Mid-low frequencies
        (freq_bins // 2, 3 * freq_bins // 4),  # Mid-high frequencies
        (3 * freq_bins // 4, freq_bins)   # High frequencies
    ]
    
    enhanced_magnitude = np.zeros_like(magnitude)
    for start, end in bands:
        band_magnitude = magnitude[start:end]
        band_noise = noise_estimate[start:end]
        
        # Apply band-specific processing
        alpha = alpha_low if start < freq_bins // 2 else alpha_high
        enhanced_band = np.maximum(band_magnitude - alpha * band_noise, beta * band_magnitude)
        enhanced_magnitude[start:end] = enhanced_band
    
    # 3. Apply spectral smoothing with adaptive window size
    window_size = (3, 3)
    enhanced_magnitude = median_filter(enhanced_magnitude, size=window_size)
    
    # 4. Apply Wiener filtering with adaptive parameters
    noise_power = np.square(noise_estimate)
    signal_power = np.maximum(np.square(enhanced_magnitude) - noise_power, 0)
    wiener_gain = signal_power / (signal_power + noise_power + 1e-10)
    enhanced_magnitude *= wiener_gain
    
    # 5. Apply adaptive spectral gating
    threshold = threshold_factor * noise_estimate
    gate_factor = np.where(enhanced_magnitude > threshold, 1.0, 0.1)
    enhanced_magnitude *= gate_factor
    
    # 6. Apply temporal smoothing with voice activity detection if available
    if WEBRTCVAD_AVAILABLE and vad:
        frames = list(frame_generator(30, audio, sr))
        vad_decisions = [is_speech(frame, vad, sr) for frame in frames]
        vad_decisions = np.array(vad_decisions)
        
        # Apply stronger smoothing to non-speech regions
        kernel_size = np.where(vad_decisions, (1, 3), (1, 7))
        enhanced_magnitude = signal.medfilt2d(enhanced_magnitude, kernel_size=kernel_size)
    else:
        # Default smoothing if VAD is not available
        enhanced_magnitude = signal.medfilt2d(enhanced_magnitude, kernel_size=(1, 5))
    
    # Reconstruct the signal
    enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
    enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length)
    
    # 7. Apply adaptive post-filtering
    # Dynamic band-pass filter based on speech content
    nyquist = sr / 2
    if WEBRTCVAD_AVAILABLE and vad and np.mean(vad_decisions) > 0.5:  # If mostly speech
        low, high = 80 / nyquist, 3000 / nyquist
    else:  # If mostly noise or VAD not available
        low, high = 100 / nyquist, 2500 / nyquist
    
    b, a = signal.butter(4, [low, high], btype='band')
    enhanced_audio = signal.filtfilt(b, a, enhanced_audio)
    
    # 8. Apply adaptive noise reduction
    enhanced_audio = signal.wiener(enhanced_audio, mysize=5)
    
    # 9. Quality preservation
    # Preserve original signal characteristics
    original_energy = np.mean(np.square(audio))
    enhanced_energy = np.mean(np.square(enhanced_audio))
    energy_ratio = np.sqrt(original_energy / (enhanced_energy + 1e-10))
    enhanced_audio *= energy_ratio
    
    # 10. Final normalization with peak preservation
    enhanced_audio = librosa.util.normalize(enhanced_audio, norm=np.inf, axis=0)
    
    return enhanced_audio

def generate_waveform_plot(audio, sr, title):
    """Generate waveform plot for audio visualization"""
    # Use a non-interactive backend to avoid threading issues
    plt.switch_backend('Agg')
    plt.figure(figsize=(10, 3))
    plt.plot(np.linspace(0, len(audio)/sr, len(audio)), audio)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Save plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_spectrogram_plot(audio, sr, title):
    """Generate spectrogram plot for audio visualization"""
    # Use a non-interactive backend to avoid threading issues
    plt.switch_backend('Agg')
    plt.figure(figsize=(10, 3))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    
    # Save plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def calculate_snr(original, enhanced):
    """Calculate Signal-to-Noise Ratio between original and enhanced audio"""
    # Ensure both arrays have the same length
    min_length = min(len(original), len(enhanced))
    original = original[:min_length]
    enhanced = enhanced[:min_length]
    
    noise = original - enhanced
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def calculate_spectral_flatness(audio, sr, n_fft=2048, hop_length=512):
    """Calculate spectral flatness of the audio"""
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    flatness = librosa.feature.spectral_flatness(S=S)
    return np.mean(flatness)

def process_audio(input_path, output_path, mode='moderate'):
    """Process audio file with error handling and progress tracking"""
    try:
        # Load audio file
        audio, sr = librosa.load(input_path, sr=16000)
        
        # Generate visualizations for original audio
        waveform_plot = generate_waveform_plot(audio, sr, "Original Waveform")
        spectrogram_plot = generate_spectrogram_plot(audio, sr, "Original Spectrogram")
        
        # Apply spectral subtraction with selected mode
        enhanced_audio = spectral_subtraction(audio, mode=mode)
        
        # Generate visualizations for enhanced audio
        enhanced_waveform = generate_waveform_plot(enhanced_audio, sr, "Enhanced Waveform")
        enhanced_spectrogram = generate_spectrogram_plot(enhanced_audio, sr, "Enhanced Spectrogram")
        
        # Save enhanced audio
        sf.write(output_path, enhanced_audio, sr)
        
        # Calculate quality metrics
        metrics = {}
        
        # Try to calculate PESQ and STOI if available
        if PESQ_AVAILABLE:
            try:
                metrics['pesq_score'] = pesq.pesq(sr, audio, enhanced_audio, 'wb')
                metrics['stoi_score'] = pystoi.stoi(audio, enhanced_audio, sr, extended=False)
            except Exception as e:
                app.logger.warning(f"Error calculating PESQ/STOI: {str(e)}")
        
        # Calculate alternative metrics
        metrics['snr'] = calculate_snr(audio, enhanced_audio)
        metrics['spectral_flatness'] = calculate_spectral_flatness(enhanced_audio, sr)
        
        return True, "File processed successfully", {
            'waveform_plot': waveform_plot,
            'spectrogram_plot': spectrogram_plot,
            'enhanced_waveform': enhanced_waveform,
            'enhanced_spectrogram': enhanced_spectrogram,
            'metrics': metrics
        }
    except Exception as e:
        return False, str(e), None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Get enhancement mode from form data
    mode = request.form.get('mode', 'moderate')
    if mode not in ['light', 'moderate', 'aggressive']:
        mode = 'moderate'
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'enhanced_{filename}')
        
        # Save uploaded file
        file.save(input_path)
        
        # Process the audio with selected mode
        success, message, visualizations = process_audio(input_path, output_path, mode=mode)
        
        if success:
            response_data = {
                'success': True,
                'message': message,
                'original_url': f'/preview/original/{filename}',
                'enhanced_url': f'/preview/enhanced/{os.path.basename(output_path)}',
                'download_url': f'/download/{os.path.basename(output_path)}',
                'visualizations': visualizations
            }
            return jsonify(response_data)
        else:
            return jsonify({'error': message}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/preview/original/<filename>')
def preview_original(filename):
    return send_file(
        os.path.join(app.config['UPLOAD_FOLDER'], filename),
        mimetype='audio/wav'
    )

@app.route('/preview/enhanced/<filename>')
def preview_enhanced(filename):
    return send_file(
        os.path.join(app.config['OUTPUT_FOLDER'], filename),
        mimetype='audio/wav'
    )

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(
        os.path.join(app.config['OUTPUT_FOLDER'], filename),
        as_attachment=True
    )

@app.route('/reprocess', methods=['POST'])
def reprocess_file():
    """Reprocess an already uploaded file with a different enhancement mode"""
    try:
        data = request.json
        if not data or 'filename' not in data or 'mode' not in data:
            return jsonify({'error': 'Missing filename or mode'}), 400
        
        filename = data['filename']
        mode = data['mode']
        
        if mode not in ['light', 'moderate', 'aggressive']:
            mode = 'moderate'
        
        # Check if file exists
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(input_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Create a new output filename with mode indicator
        base_name, ext = os.path.splitext(filename)
        output_filename = f'enhanced_{base_name}_{mode}{ext}'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Process the audio with selected mode
        success, message, visualizations = process_audio(input_path, output_path, mode=mode)
        
        if success:
            response_data = {
                'success': True,
                'message': message,
                'original_url': f'/preview/original/{filename}',
                'enhanced_url': f'/preview/enhanced/{output_filename}',
                'download_url': f'/download/{output_filename}',
                'visualizations': visualizations
            }
            return jsonify(response_data)
        else:
            return jsonify({'error': message}), 500
    except Exception as e:
        app.logger.error(f"Error in reprocess_file: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 