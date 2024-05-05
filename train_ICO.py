import os
from pyexpat import model
import librosa
import resampy
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.optimizers import Adam
import random

# Define the ICO-CE model class
class ICO_CE:
    def __init__(self, n_estimators=10, max_iter=100, learning_rate=0.01, threshold=0.5):
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.models = []
        self.weights = []

    def fit(self, X_train, y_train):
        n_samples = len(X_train)
        n_features = X_train.shape[1]
        self.models = []
        self.weights = []

        # Initialize models and weights
        for _ in range(self.n_estimators):
            model = Sequential([
                Dense(512, activation='relu', input_shape=(n_features,)),
                BatchNormalization(),
                Dropout(0.5),
                Dense(256, activation='relu'),
                BatchNormalization(),
                Dense(256, activation='relu'),
                BatchNormalization(),
                Dense(256, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(4, activation='softmax')
            ])
            model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
            model.save("ICO_model.h5")
            self.models.append(model)
            self.weights.append(1.0 / self.n_estimators)

        # Train models
        for i in range(self.n_estimators):
            model = self.models[i]
            weight = self.weights[i]
            for _ in range(self.max_iter):
                # Sample with replacement from the training data
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_batch = X_train[indices]
                y_batch = y_train[indices]
                model.fit(X_batch, y_batch, epochs=1, batch_size=32, verbose=0)
            self.weights[i] = weight * model.evaluate(X_train, y_train, verbose=0)[1]

        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [weight / total_weight for weight in self.weights]

    def predict(self, X_test):
        # Make predictions using weighted average
        predictions = np.zeros((X_test.shape[0], 4))
        for i in range(self.n_estimators):
            model = self.models[i]
            weight = self.weights[i]
            predictions += weight * model.predict(X_test)
        return predictions
    
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

# Train the ICO-CE model
ico_ce = ICO_CE(n_estimators=10, max_iter=100, learning_rate=0.01, threshold=0.5)
ico_ce.fit(X_train, y_train)

# Evaluate the model
predictions = ico_ce.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)
accuracy = np.mean(predicted_labels == true_labels)

print("Test accuracy:", accuracy)
