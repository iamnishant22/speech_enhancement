from flask import Flask, render_template, request, send_file, jsonify
import os
from werkzeug.utils import secure_filename
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import io
import base64
import gc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'flac', 'm4a'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def cleanup_files():
    """Clean up temporary files"""
    try:
        for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error in cleanup_files: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def spectral_subtraction(audio, sr=16000, n_fft=2048, hop_length=512, mode='moderate'):
    """Basic spectral subtraction with memory optimization"""
    try:
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Compute STFT
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(D)
        phase = np.angle(D)
        
        # Estimate noise floor
        noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)
        
        # Apply spectral subtraction with different modes
        if mode == 'light':
            alpha = 1.0
            beta = 0.1
        elif mode == 'moderate':
            alpha = 1.5
            beta = 0.2
        else:  # aggressive
            alpha = 2.0
            beta = 0.3
        
        # Apply spectral subtraction
        enhanced_magnitude = np.maximum(magnitude - alpha * noise_floor, beta * magnitude)
        
        # Reconstruct the signal
        enhanced_D = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_D, hop_length=hop_length)
        
        # Clean up memory
        del D, magnitude, phase, noise_floor, enhanced_magnitude, enhanced_D
        gc.collect()
        
        return enhanced_audio
    except Exception as e:
        logger.error(f"Error in spectral_subtraction: {e}")
        raise

def generate_waveform_plot(audio, sr, title):
    """Generate waveform plot"""
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(audio)/sr, len(audio)), audio)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    
    # Save plot to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_spectrogram_plot(audio, sr, title):
    """Generate spectrogram plot"""
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    # Save plot to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.read()).decode('utf-8')

def process_audio(input_path, output_path, mode='moderate'):
    """Process audio file with spectral subtraction"""
    # Load audio
    audio, sr = librosa.load(input_path, sr=None)
    
    # Process audio
    enhanced_audio = spectral_subtraction(audio, sr=sr, mode=mode)
    
    # Save enhanced audio
    sf.write(output_path, enhanced_audio, sr)
    
    # Generate visualizations
    waveform_plot = generate_waveform_plot(audio, sr, 'Original Waveform')
    spectrogram_plot = generate_spectrogram_plot(audio, sr, 'Original Spectrogram')
    enhanced_waveform = generate_waveform_plot(enhanced_audio, sr, 'Enhanced Waveform')
    enhanced_spectrogram = generate_spectrogram_plot(enhanced_audio, sr, 'Enhanced Spectrogram')
    
    return {
        'visualizations': {
            'waveform_plot': waveform_plot,
            'spectrogram_plot': spectrogram_plot,
            'enhanced_waveform': enhanced_waveform,
            'enhanced_spectrogram': enhanced_spectrogram
        }
    }

@app.route('/')
def index():
    try:
        cleanup_files()  # Clean up old files on startup
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error in index route: {e}")
        return "An error occurred", 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'enhanced_{filename}')
        
        file.save(input_path)
        
        try:
            process_audio(input_path, output_path)
            return jsonify({
                'success': True,
                'filename': filename,
                'message': 'File processed successfully'
            })
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return jsonify({'error': 'Error processing file'}), 500
        finally:
            # Clean up input file after processing
            try:
                os.remove(input_path)
            except:
                pass
    except Exception as e:
        logger.error(f"Error in upload_file: {e}")
        return jsonify({'error': 'Server error'}), 500

@app.route('/preview/original/<filename>')
def preview_original(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/preview/enhanced/<filename>')
def preview_enhanced(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename))

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename), as_attachment=True)

@app.route('/reprocess', methods=['POST'])
def reprocess_file():
    data = request.get_json()
    if not data or 'filename' not in data:
        return jsonify({'success': False, 'error': 'No filename provided'})
    
    filename = data['filename']
    mode = data.get('mode', 'moderate')
    
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'enhanced_{filename}')
    
    if not os.path.exists(input_path):
        return jsonify({'success': False, 'error': 'Original file not found'})
    
    # Process audio
    result = process_audio(input_path, output_path, mode)
    
    return jsonify({
        'success': True,
        'message': 'File reprocessed successfully',
        'enhanced_url': f'/preview/enhanced/enhanced_{filename}',
        'download_url': f'/download/enhanced_{filename}',
        'visualizations': result['visualizations']
    })

if __name__ == '__main__':
    app.run(debug=True) 