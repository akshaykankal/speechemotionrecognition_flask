from urllib import response
from flask import Flask, request, jsonify
import numpy as np
import os
import librosa
from keras.models import load_model
from flask_cors import CORS
import wave
from datetime import datetime

app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

CORS(app)  # Enable CORS for all routes

# Load the trained model
model = load_model('ICO_model.h5')


# Compile the model with the same metrics as it was compiled with during training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Function to extract features from audio file
def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    result = np.array([])
    if chroma:
        chroma = librosa.feature.chroma_stft(y=X, sr=sample_rate)
        result = np.hstack((result, np.mean(chroma.T, axis=0)))
    if mfcc:
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
        result = np.hstack((result, np.mean(mfccs.T, axis=0)))
    if mel:
        mel = librosa.feature.melspectrogram(y=X, sr=sample_rate)
        result = np.hstack((result, np.mean(mel.T, axis=0)))
    return result

# Route for predicting emotions
@app.route('/predict', methods=['POST'])
def predict_emotion():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})
    
    file = request.files['file']
    emotion = request.form.get('emotion')
    
    # If the user does not select a file, the browser may send an empty file without a filename
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})
    
    # Save the uploaded file to a temporary location with WAV format
    temp_dir = "./"
    temp_file_path = os.path.join(temp_dir, f"{emotion}_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav")
    
    if file.filename.lower().endswith('.wav'):
        file.save(temp_file_path)
    else:
        with wave.open(temp_file_path, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(44100)  # Sample rat`e
            wf.writeframes(file.read())
    
    # Extract features from the audio file
    features = extract_features(temp_file_path)

    # Make prediction
    prediction = model.predict(np.expand_dims(features, axis=0))
    emotion_labels = ['happy', 'anger', 'sadness', 'neutral']
    predicted_emotion = emotion_labels[np.argmax(prediction)]


    return jsonify({'success': True, 'emotion': predicted_emotion})

if __name__ == '__main__':
    # Run the Flask app on host 0.0.0.0 and port 10000
    app.run(host='0.0.0.0', port=10000)