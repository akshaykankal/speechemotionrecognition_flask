import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

# Step 1: Preprocessing
def preprocess_data(X, y):
    # Reshape X to collapse the time dimension
    X_flat = X.reshape(X.shape[0], -1)

    # Perform class imbalance processing using SMOTE-ENC
    smote_enc = SMOTE(sampling_strategy='auto', k_neighbors=5, n_jobs=-1)
    X_resampled, y_resampled = smote_enc.fit_resample(X_flat, y)

    # Reshape X_resampled back to the original shape
    num_samples = X_resampled.shape[0]
    X_resampled_reshaped = X_resampled.reshape(num_samples, X.shape[1], X.shape[2])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled_reshaped, y_resampled, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Function to extract features
def extract_features(audio_file, max_length):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)

    # Pad or truncate features to max_length
    features = []
    for feature in [mfccs, chroma, spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_rolloff, zero_crossing_rate]:
        if feature.shape[1] >= max_length:
            features.append(feature[:, :max_length])
        else:
            features.append(np.pad(feature, ((0, 0), (0, max_length - feature.shape[1])), mode='constant'))

    return np.vstack(features)

# Load data
data_dir = 'dataset/Dataset'
max_length = 100  # Maximum length of features
labels = {'anger': 0, 'happy': 1, 'sadness': 2, 'neutral': 3}
X, y = [], []

for emotion in labels.keys():
    emotion_dir = os.path.join(data_dir, emotion)
    for filename in os.listdir(emotion_dir):
        if filename.endswith('.wav'):
            audio_file = os.path.join(emotion_dir, filename)
            X.append(extract_features(audio_file, max_length))
            y.append(labels[emotion])

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Preprocess data
X_train, X_test, y_train, y_test = preprocess_data(X, y)

# Build and train classification model
model = Sequential([
    LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[EarlyStopping(patience=3)])

# Save the trained model
model.save('paper_model.h5')

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, np.argmax(y_pred, axis=1))
print("Test accuracy:", accuracy)