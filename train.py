import os
import librosa
import resampy
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
import random

# Function to extract features from audio files
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

# Define directories
happy_dir = "/content/Dataset/happy"
anger_dir = "/content/Dataset/anger"
sadness_dir = "/content/Dataset/sadness"
neutral_dir = "/content/Dataset/neutral"

# Load data and extract features
X, y = [], []
for dir_path, dir_names, file_names in os.walk("Dataset"):
    for file_name in file_names:
        if file_name.endswith(".wav"):
            file_path = os.path.join(dir_path, file_name)
            emotion = dir_path.split("/")[-1]  # Extracting emotion from directory path
            feature = extract_features(file_path)
            X.append(feature)
            y.append(emotion)

# Convert data to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert emotion labels to integers using label mapping
label_map = {"happy": 0, "anger": 1, "sadness": 2, "neutral": 3}
y_train_int = np.array([label_map[label] for label in y_train])
y_test_int = np.array([label_map[label] for label in y_test])

# Convert integer labels to one-hot encoded vectors
y_train = to_categorical(y_train_int, num_classes=4)
y_test = to_categorical(y_test_int, num_classes=4)

# Data augmentation for audio data
def augment_data(X, max_length):
    augmented_X = []
    for audio in X:
        # Randomly apply time stretching
        stretch_factor = random.uniform(0.8, 1.2)
        augmented_audio = librosa.effects.time_stretch(audio, rate=stretch_factor)
        
        # Pad or trim the augmented audio to the maximum length
        if len(augmented_audio) < max_length:
            # Pad the audio signal with zeros
            augmented_audio = np.pad(augmented_audio, (0, max_length - len(augmented_audio)))
        elif len(augmented_audio) > max_length:
            # Trim the audio signal
            augmented_audio = augmented_audio[:max_length]
        
        augmented_X.append(augmented_audio)
    return np.array(augmented_X)




# Determine the maximum length of audio signals in the dataset
max_length = max(len(audio) for audio in X_train)

# Augment training data
X_train_augmented = augment_data(X_train, max_length)

# Combine original and augmented data
X_train_combined = np.vstack((X_train, X_train_augmented))
y_train_combined = np.vstack((y_train, y_train))

# Define the deep learning model
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_combined, y_train_combined, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Save the model
model.save("speech_emotion_model.h5")
