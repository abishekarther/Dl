ex 2
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
X, y = fetch_california_housing(return_X_y=True)

# Preprocessing
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(8,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Evaluate
loss, mae = model.evaluate(X_test, y_test)
print("Test MAE:", mae)

# Predict
print("Predicted price:", model.predict(X_test[:1]))


ex 4

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.utils import load_img, img_to_array

# Load pretrained VGG16
model = VGG16(weights='imagenet')

# Load image (keep dog.jpg in notebook folder)
img = load_img('Dog.jpg', target_size=(224, 224))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Predict
pred = model.predict(x)
print(decode_predictions(pred, top=3)[0])
