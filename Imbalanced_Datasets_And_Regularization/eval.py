import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMG_SIZE = 128
DATA_DIR = "hw2_data"

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

model = tf.keras.models.load_model("best_model.h5")
_, acc = model.evaluate(X, y, verbose=0)
print("Evaluation accuracy:", acc)