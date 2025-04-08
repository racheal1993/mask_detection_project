import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Corrected dataset paths
with_mask_path = os.path.join("data", "with_mask")
without_mask_path = os.path.join("data", "without_mask")

data = []
labels = []

# Load images with mask
for img_name in os.listdir(with_mask_path):
    img_path = os.path.join(with_mask_path, img_name)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (100, 100))
        data.append(img)
        labels.append(0)  # 0 = with mask

# Load images without mask
for img_name in os.listdir(without_mask_path):
    img_path = os.path.join(without_mask_path, img_name)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (100, 100))
        data.append(img)
        labels.append(1)  # 1 = without mask

# Preprocessing
data = np.array(data) / 255.0
labels = to_categorical(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the model
model.save("mask_detector_model.h5")
print("✅ Model training complete and saved as mask_detector_model.h5")

