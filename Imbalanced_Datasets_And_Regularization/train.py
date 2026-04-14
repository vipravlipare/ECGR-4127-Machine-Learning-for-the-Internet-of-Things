import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMG_SIZE = 128
DATA_DIR = "hw2_data"  # your 10-image dataset

# Load images
def load_data():
    X, y = [], []
    for label, cls in enumerate(["negative", "positive"]):
        folder = os.path.join(DATA_DIR, cls)
        for fname in os.listdir(folder):
            img = load_img(os.path.join(folder, fname), target_size=(IMG_SIZE, IMG_SIZE))
            X.append(img_to_array(img) / 255.0)
            y.append(label)
    return np.array(X), np.array(y)

X, y = load_data()

# Build a small CNN like your CIFAR example
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X, y, epochs=5, batch_size=2, verbose=2) 

# Save the model
model.save("best_model.h5")
print("Model saved as best_model.h5")