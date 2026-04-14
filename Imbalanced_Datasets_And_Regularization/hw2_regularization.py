import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    Dropout(0.5),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu',),
    Dropout(0.5),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu',),
    Dropout(0.5),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=12,
    batch_size=64,
    validation_data=(x_test, y_test),
    verbose=2
)

val_loss, val_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Validation Accuracy: {val_acc}")